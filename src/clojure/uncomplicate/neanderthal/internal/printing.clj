;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.printing
  (:require [clojure.pprint :refer [cl-format]]
            [uncomplicate.fluokitten.core :refer [op join]]
            [uncomplicate.neanderthal
             [math :refer [ceil]]
             [core :refer [rows dia submatrix subvector mrows ncols dim entry]]]
            [uncomplicate.neanderthal.internal.api :refer :all])
  (:import [uncomplicate.neanderthal.internal.api Matrix LayoutNavigator Region DenseStorage FullStorage
            Changeable Matrix Vector]));;TODO clean up

;; ====================================================================



(defn ^:private unsupported []
  (throw (UnsupportedOperationException. (format "This operation is not supported in wrappers."))))

(def ^:private compile-format #'clojure.pprint/compile-format)

(def format-g (compile-format "~6,2,,1G"))
(def format-f (compile-format "~7F"))
(def format-a (compile-format "~4@A~8T"))
(def format-seq (compile-format "~{~8A~}"))
(def format-band-col (compile-format "~&~3a~4@T~{~8a~}~a~%"))
(def format-band-dia (compile-format "~&~3a~4@T~{~7a~}~a~%"))
(def format-band (compile-format "~&~4a~{~8a~}~%"))

(defn string-table [^long m ^long n]
  (let [st ^objects (make-array String (+ m 2) (+ n 2))
        header ^objects (aget st 0)
        footer ^objects (aget st (inc m))]
    (dotimes [j (+ n 2)]
      (aset header j ""))
    (aset header 0 (cl-format nil format-a "\u250f"))
    (aset header (inc n) (cl-format nil format-a "\u2513"))
    (dotimes [i m]
      (let [r ^objects (aget st (inc i))]
        (aset r 0 "")
        (dotimes [j (+ n 2)]
          (aset r j (cl-format nil format-a "*")))
        (aset r 0 "")
        (aset r (inc n) "")))
    (dotimes [j (+ n 2)]
      (aset footer j ""))
    (aset footer 0 (cl-format nil format-a "\u2517"))
    (aset footer (inc n) (cl-format nil format-a "\u2518"))
    st))

(defn aset-table [^objects st ^long i ^long j val]
  (aset st i j (cl-format nil format-a val)))

(defn format-row [^java.io.Writer w s-left s-right]
  (when (seq s-left) (cl-format w format-seq s-left))
  (when (and (seq s-left) (seq s-right)) (cl-format w format-a "\u22ef"))
  (when (seq s-right)) (cl-format w format-seq s-right))

(let [default-settings {:matrix-width 5
                        :matrix-height 5}
      settings (atom default-settings)
      pad-str (cl-format nil format-a "*")
      flip-op (fn [x y] (op y x))]

  (defn printer-settings!
    ([new]
     (swap! settings into new))
    ([]
     (reset! settings default-settings)))

  (defn printer-settings []
    @settings)

  (defn print-vector
    ([^java.io.Writer w formatter x]
     (when (< 0 (dim x))
       (let [print-width (min (dim x) (long (:matrix-width @settings)))
             print-left (long (if (= print-width (dim x)) print-width (ceil (/ print-width 2))))
             print-right (- print-width print-left)]
         (format-row w (map formatter (seq (subvector x 0 print-left)))
                     (map formatter (seq (subvector x (- (dim x) print-right) print-right)))))))
    ([^java.io.Writer w x]
     (let [max-value (double (amax (engine x) x))
           min-value (entry x (iamin (engine x) x))
           formatter (partial cl-format nil
                              (if (and (not (< 0.0 min-value 0.01)) (< max-value 10000.0))
                                format-f
                                format-g))]
       (.write w "\n[")
       (print-vector w formatter x)
       (.write w "]\n"))))

  (defn print-ge
    ([^java.io.Writer w formatter a]
     (when (< 0 (dim a))
       (let [print-height (min (mrows a) (long (:matrix-height @settings)))
             print-width (min (ncols a) (long (:matrix-width @settings)))
             print-top (long (if (= print-height (mrows a)) print-height (ceil (/ print-height 2))))
             print-bottom (long (- print-height print-top))]
         (doseq [r (rows (submatrix a 0 0 print-top (ncols a)))]
           (print-vector w formatter r)
           (.write w "\n"))
         (when (< 0 print-bottom)
           (cl-format w format-seq (cons "" (repeat print-width (cl-format nil format-a "\u22ee"))))
           (.write w "\n")
           (doseq [r (rows (submatrix a (- (mrows a) print-bottom) 0 print-bottom (ncols a)))]
             (print-vector w formatter r)
             (.write w "\n"))))))
    ([^java.io.Writer w a]
     (when (< 0 (dim a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil  (if (< max-value 10000.0) format-f format-g))]
         (.write w (format "\n\u25a5\n") )
         (print-ge w formatter a)))))

  (defn print-uplo
    ([^java.io.Writer w formatter a]
     (when (< 0 (dim a))
       (let [reg (region a)
             print-height (min (mrows a) (long (:matrix-height @settings)))
             print-width (min (ncols a) (long (:matrix-width @settings)))
             sub-a (submatrix a 0 0 print-height print-width)
             op-fn (if (.isLower reg) op flip-op)]
         (.write w "\n\u25a5")
         (cl-format w format-seq (repeat print-width  ""))
         (.write w (if (= print-width (ncols a)) "\u2513\n" "\u22ef\u2513\n"))
         (doseq [r (rows sub-a)]
           (cl-format w format-seq
                      (op-fn (map formatter (seq r)) (repeat (- print-height (dim r)) pad-str)))
           (.write w "\n"))
         (.write w (if (= print-height (mrows a)) "\u2517" " \u22ee\n\u2517")))))
    ([^java.io.Writer w a]
     (when (< 0 (dim a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil (if (< max-value 10000.0) format-f format-g))]
         (print-uplo w formatter a)))))

  (defn print-banded
    ([^java.io.Writer w formatter a]
     (let [stor (storage a)
           reg (region a)
           ku (.ku reg)
           width (.fd stor)
           height (.sd ^FullStorage stor)
           print-width (min width (long (:matrix-width @settings)))
           print-height (min height (long (:matrix-height @settings)))
           print-table (string-table print-height print-width)
           k-max (min ku (long (/ print-height 2)))]
       (dotimes [i print-height]
         (let [k (- k-max i)
               d (dia a k)
               j0 (max 0 k)]
           (dotimes [j (min (dim d) print-width (- print-width k))]
             (aset print-table (inc i) (inc (+ j0 j)) (formatter (entry d j))))))
       (doseq [print-row print-table]
         (cl-format w format-seq print-row)
         (.write w "\n"))))
    ([^java.io.Writer w a]
     (when (< 0 (dim a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil (if (< max-value 10000.0) format-f format-g))]
         (print-banded w formatter a)))))

  (defn print-packed
    ([^java.io.Writer w formatter ^Matrix a]
     (let [nav (navigator a)
           stor (storage a)]
       (dotimes [j (.fd stor)]
         (print-vector w formatter (.stripe nav a j))
         (.write w "\n"))))
    ([^java.io.Writer w a]
     (let [max-value (double (amax (engine a) a))
           formatter (partial cl-format nil  (if (< max-value 10000.0) format-f format-g))]
       (print-packed w formatter a)))))
