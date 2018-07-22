;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.printing
  (:require [clojure.pprint :refer [cl-format]]
            [uncomplicate.neanderthal
             [math :refer [ceil]]
             [core :refer [subvector mrows ncols dim entry dia]]]
            [uncomplicate.neanderthal.internal.api :refer :all])
  (:import [uncomplicate.neanderthal.internal.api LayoutNavigator Region DenseStorage FullStorage]))

;; ====================================================================

(def ^:private compile-format #'clojure.pprint/compile-format)

(def ^:const format-g (compile-format "~6,2,,1G"))
(def ^:const format-f (compile-format "~7,2F"))
(def ^:const format-a (compile-format "~4@A~8T"))
(def ^:const format-a7 (compile-format "~4@A~7T"))
(def ^:const format-a8 (compile-format "~4@A~8T"))
(def ^:const format-seq (compile-format "~{~8A~}"))
(def ^:const format-header-row (compile-format "~{~7A~}~%"))
(def ^:const format-header-col (compile-format "~{~8A~}~%"))

(let [default-settings {:matrix-width 5
                        :matrix-height 5}
      settings (atom default-settings)
      pad-str (cl-format nil format-a "*")
      pad-dot (cl-format nil format-a ".")
      diag-arrow (cl-format nil format-a \u2198)
      row-arrow (cl-format nil format-a7 \u2192)
      col-arrow (cl-format nil format-a8 \u2193)
      row-major (cl-format nil format-a \u25a4)
      col-major (cl-format nil format-a \u25a5)
      dia-major (cl-format nil format-a \u25a7)
      hdots (cl-format nil format-a7 \u22ef)
      vdots (cl-format nil format-a8 \u22ee)
      up-elipsis (cl-format nil format-a8 \u22f0)
      down-elipsis (cl-format nil format-a8 \u22f1)
      hline (cl-format nil format-a "\u2500")
      vline (cl-format nil format-a "|")
      nw-corner (cl-format nil format-a "\u250f")
      ne-corner (cl-format nil format-a "\u2513")
      sw-corner (cl-format nil format-a "\u2517")
      se-corner (cl-format nil format-a "\u251b")
      five-dots (cl-format nil format-a "\u2059")
      one (cl-format nil format-a "\u00b71\u00b7")]

  (defn printer-settings!
    ([new]
     (swap! settings into new))
    ([]
     (reset! settings default-settings)))

  (defn printer-settings []
    @settings)

  (defn string-table
    ([^long m ^long n placeholder]
     (let [st ^objects (make-array String (+ m 2) (+ n 2))
           header ^objects (aget st 0)
           footer ^objects (aget st (inc m))]
       (dotimes [j (+ n 2)]
         (aset header j ""))
       (aset header 0 nw-corner)
       (aset header (inc n) ne-corner)
       (dotimes [i m]
         (let [r ^objects (aget st (inc i))]
           (aset r 0 "")
           (dotimes [j (+ n 2)]
             (aset r j placeholder))
           (aset r 0 "")
           (aset r (inc n) "")))
       (dotimes [j (+ n 2)]
         (aset footer j ""))
       (aset footer 0 sw-corner)
       (aset footer (inc n) se-corner)
       st))
    ([^long m ^long n]
     (string-table m n (cl-format nil format-a "*"))))

  (defn format-row [^java.io.Writer w s-left s-right]
    (when (seq s-left) (cl-format w format-seq s-left))
    (when (and (seq s-left) (seq s-right)) (cl-format w hdots))
    (when (seq s-right)) (cl-format w format-seq s-right))

  (defn print-vector
    ([^java.io.Writer w formatter x]
     (when (< 0 (dim x))
       (let [print-width (min (dim x) (long (:matrix-width @settings)))
             print-left (long (if (= print-width (dim x)) print-width (ceil (/ print-width 2))))
             print-right (- print-width print-left)]
         (format-row w (map formatter (seq (subvector x 0 print-left)))
                     (map formatter (seq (subvector x (- (dim x) print-right) print-right)))))))
    ([^java.io.Writer w x]
     (when (< 0 (dim x))
       (.write w "\n[")
       (let [max-value (double (amax (engine x) x))
             min-value (entry x (iamin (engine x) x))
             formatter (partial cl-format nil
                                (if (and (not (< 0.0 min-value 0.01)) (< max-value 10000.0))
                                  format-f
                                  format-g))]

         (print-vector w formatter x))
       (.write w "]\n"))))

  (defn print-ge
    ([^java.io.Writer w formatter a]
     (when (< 0 (dim a))
       (let [m (mrows a)
             n (ncols a)
             print-height (let [h (min m (long (:matrix-height @settings)))]
                            (long (if (= m h) h (+ h (- 1 (long (mod h 2)))))))
             print-chunk-h (long (if (= m print-height) (ceil (/ print-height 2)) (/ print-height 2)))
             print-width (let [w (min n (long (:matrix-width @settings)))]
                           (long (if (= n w) w (+ w (- 1 (long (mod w 2)))))))
             print-chunk-w (long (if (= n print-width) (ceil (/ print-width 2)) (/ print-width 2)))
             print-table (string-table print-height print-width five-dots)]
         (dotimes [i print-chunk-h]
           (dotimes [j print-chunk-w]
             (aset print-table (inc i) (inc j) (formatter (entry a i j)))
             (aset print-table (inc i) (- print-width j) (formatter (entry a i (dec (- n j)))))
             (aset print-table (- print-height i) (inc j) (formatter (entry a (dec (- m i)) j)))
             (aset print-table (- print-height i) (- print-width j)
                   (formatter (entry a (dec (- m i)) (dec (- n j)))))))
         (.write w "\n")
         (aset print-table 0 0 (if (.isColumnMajor (navigator a)) col-major row-major))
         (dotimes [i print-height]
           (aset print-table (inc i) 0 row-arrow))
         (dotimes [j print-width]
           (aset print-table 0 (inc j) col-arrow))
         (cl-format w format-header-col (first print-table))
         (doseq [print-row (rest print-table)]
           (cl-format w format-seq print-row)
           (.write w "\n")))))
    ([^java.io.Writer w a]
     (when (< 0 (dim a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil  (if (< max-value 10000.0) format-f format-g))]
         (print-ge w formatter a)))))

  (defn print-uplo
    ([^java.io.Writer w formatter a placeholder]
     (when (< 0 (dim a))
       (let [reg (region a)
             nav (navigator a)
             print-height (min (mrows a) (long (:matrix-height @settings)))
             print-width (min (ncols a) (long (:matrix-width @settings)))
             print-table (string-table print-height print-width (cl-format nil format-a placeholder))]
         (dotimes [i print-height]
           (loop [j (.rowStart reg i)]
             (when (< j (min print-width (.rowEnd reg i)))
               (aset print-table (inc i) (inc j) (formatter (entry a i j)))
               (recur (inc j)))))
         (when (.isDiagUnit reg)
           (dotimes [j (min print-width print-height)]
             (aset print-table (inc j) (inc j) one)))
         (.write w "\n")
         (aset print-table 0 0 (if (.isColumnMajor nav) col-major row-major))
         (when (< print-height (mrows a))
           (aset print-table (inc print-height) 0 vline)
           (aset print-table 1 (inc print-width) hdots)
           (aset print-table (inc print-height) (inc print-width) vline))
         (when (< print-width (ncols a))
           (aset print-table 0 (inc print-width) hline)
           (aset print-table (inc print-height) 1 vdots)
           (aset print-table (inc print-height) (inc print-width) hline))
         (when (and (< print-height (mrows a)) (< print-width (ncols a)))
           (aset print-table (inc print-height) (inc print-width) down-elipsis))
         (dotimes [i print-height]
           (aset print-table (inc i) 0 row-arrow))
         (dotimes [j print-width]
           (aset print-table 0 (inc j) col-arrow))
         (cl-format w format-header-col (first print-table))
         (doseq [print-row (rest print-table)]
           (cl-format w format-seq print-row)
           (.write w "\n")))))
    ([^java.io.Writer w a placeholder]
     (when (< 0 (dim a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil (if (< max-value 10000.0) format-f format-g))]
         (print-uplo w formatter a placeholder)))))

  (defn print-banded
    ([^java.io.Writer w formatter a]
     (let [nav (navigator a)
           stor (storage a)
           reg (region a)
           ku (max 0 (.ku reg))
           width (.fd stor)
           height (.sd ^FullStorage stor)
           print-height (min height (long (:matrix-height @settings)))
           print-width (min width (long (:matrix-width @settings)))
           print-table (string-table print-height print-width)
           k-max (min ku (max (long (/ print-height 2)) (- print-height (max 0 (.kl reg)))))
           format-header (if (.isColumnMajor nav) format-header-col format-header-row)
           direction-arrow (if (.isColumnMajor nav) col-arrow row-arrow)]
       (dotimes [i print-height]
         (let [k (- k-max i)
               d (dia a k)
               j0 (max 0 k)]
           (if (< 0 (dim d))
             (dotimes [j (min (dim d) print-width (- print-width k))]
               (aset print-table (inc i) (inc (+ j0 j)) (formatter (entry d j))))
             (dotimes [j print-width]
               (aset print-table (inc i) (inc j) one)))))
       (aset print-table 0 0 (if (.isColumnMajor nav) col-major row-major))
       (dotimes [i print-height]
         (aset print-table (inc i) 0 diag-arrow))
       (dotimes [j print-width]
         (aset print-table 0 (inc j) direction-arrow))
       (if (< k-max ku)
         (aset print-table 0 (inc print-width) (if (< print-width width) up-elipsis vline))
         (when (< print-width width)
           (aset print-table 0 (inc print-width) hline)
           (aset print-table 1 (inc print-width) hdots)))
       (when (< print-height (+ k-max (.kl reg) 1))
         (aset print-table (inc print-height) 0 vline)
         (aset print-table (inc print-height) 1 vdots)
         (if (< print-width width)
           (aset print-table (inc print-height) (inc print-width) down-elipsis)
           (aset print-table (inc print-height) (inc print-width) vline)))
       (.write w "\n")
       (cl-format w format-header (first print-table))
       (doseq [print-row (rest print-table)]
         (cl-format w format-seq print-row)
         (.write w "\n"))))
    ([^java.io.Writer w a]
     (when (< 0 (dim a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil (if (< max-value 10000.0) format-f format-g))]
         (print-banded w formatter a)))))

  (defn print-diagonal
    ([^java.io.Writer w formatter a]
     (let [stor (storage a)
           reg (region a)
           ku (.ku reg)
           kl (.kl reg)
           width (ncols a)
           height (inc (+ ku kl))
           print-height (min height (long (:matrix-height @settings)))
           print-width (min width (long (:matrix-width @settings)))
           print-table (string-table print-height print-width pad-dot)
           k-max (min ku (max (long (/ print-height 2)) (- print-height kl)))
           format-header format-header-col]
       (dotimes [i print-height]
         (let [k (- k-max i)
               d (dia a k)
               j0 (max 0 k)]
           (dotimes [j (min (dim d) print-width (- print-width k))]
             (aset print-table (inc i) (inc (+ j0 j)) (formatter (entry d j))))))
       (aset print-table 0 0 dia-major)
       (dotimes [i print-height]
         (aset print-table (inc i) 0 diag-arrow))
       (if (< k-max ku)
         (aset print-table 0 (inc print-width) (if (< print-width width) up-elipsis vline))
         (when (< print-width width)
           (aset print-table 0 (inc print-width) hline)
           (aset print-table 1 (inc print-width) hdots)))
       (when (< print-height (+ k-max (.kl reg) 1))
         (aset print-table (inc print-height) 0 vline)
         (aset print-table (inc print-height) 1 vdots)
         (if (< print-width width)
           (aset print-table (inc print-height) (inc print-width) down-elipsis)
           (aset print-table (inc print-height) (inc print-width) vline)))
       (.write w "\n")
       (cl-format w format-header (first print-table))
       (doseq [print-row (rest print-table)]
         (cl-format w format-seq print-row)
         (.write w "\n"))))
    ([^java.io.Writer w a]
     (when (< 0 (dim a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil (if (< max-value 10000.0) format-f format-g))]
         (print-diagonal w formatter a))))))
