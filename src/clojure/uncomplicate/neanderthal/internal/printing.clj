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
             [block :refer [ecount lower? column?]]
             [core :refer [rows dias submatrix subvector mrows ncols dim entry]]]
            [uncomplicate.neanderthal.internal.api :refer :all])
  (:import [uncomplicate.neanderthal.internal.api UploNavigator BandNavigator BandedMatrix Matrix
            LayoutNavigator]));;TODO clean up

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

(defn format-row [^java.io.Writer w s-left s-right]
  (when (seq s-left) (cl-format w format-seq s-left))
  (when (and (seq s-left) (seq s-right)) (cl-format w format-a "\u22ef"))
  (when (seq s-right)) (cl-format w format-seq s-right))

;; ===================== Banded Matrix Printing ====================================================

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
     (when (< 0 (ecount a))
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
     (when (< 0 (ecount a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil  (if (< max-value 10000.0) format-f format-g))]
         (.write w (format "\n%s\n" (if (column? a) "\u25a5" "\u25a4")) )
         (print-ge w formatter a)))))

  (defn print-uplo
    ([^java.io.Writer w formatter a]
     (when (< 0 (ecount a))
       (let [print-height (min (mrows a) (long (:matrix-height @settings)))
             print-width (min (ncols a) (long (:matrix-width @settings)))
             sub-a (submatrix a 0 0 print-height print-width)
             op-fn (if (lower? a) op flip-op)]
         (.write w (if (column? a) "\n\u25a5" "\n\u25a4"))
         (cl-format w format-seq (repeat print-width  ""))
         (.write w (if (= print-width (ncols a)) "\u2513\n" "\u22ef\u2513\n"))
         (doseq [r (rows sub-a)]
           (cl-format w format-seq
                      (op-fn (map formatter (seq r)) (repeat (- print-height (dim r)) pad-str)))
           (.write w "\n"))
         (.write w (if (= print-height (mrows a)) "\u2517" " \u22ee\n\u2517")))))
    ([^java.io.Writer w a]
     (when (< 0 (ecount a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil (if (< max-value 10000.0) format-f format-g))]
         (print-uplo w formatter a)))))

  (defn print-banded
    ([^java.io.Writer w formatter ^BandedMatrix a]
     (let [band-nav (band-navigator a)
           m (.mrows a)
           n (.ncols a)
           kl (.kl a)
           ku (.ku a)
           width (.width band-nav m n kl ku)
           height (.height band-nav m n kl ku)
           print-width (min width (long (:matrix-width @settings)))
           print-height (min height (long (:matrix-height @settings)))
           kd (.kd band-nav kl ku)
           full-width (= print-width width)
           full-height (= print-height height)
           vectors (if (column? a) dias rows)]
       (if (column? a)
         (let [first-dia (max 0 (- kd (long (/ print-height 2))))]
           (cl-format w format-band-col "\u25a5" (repeat print-width "\u2193")
                      (if full-width "\u2513" "\u22ef\u2513"))
           (when (< 0 first-dia) (.write w " \u22ee\n"))
           (loop [i kd ds (take print-height (drop first-dia (vectors a)))]
             (when ds
               (let [pad (- (max i 0) first-dia)
                     entries (take (- print-width (max 0 pad)) (first ds))]
                 (cl-format w format-band "\u2198"
                            (op (repeat pad pad-str) (map formatter entries)
                                (repeat (- print-width (+ (max 0 pad) (count entries))) pad-str))))
               (recur (dec i) (next ds)))))
         (let [first-col (max 0 (- kd (long (/ print-width 2))))]
           (cl-format w format-band-dia (if (= 0 first-col) "\u25a4" "\u25a4 \u22ef")
                      (repeat print-width  "\u2198") (if full-width "\u2513" "\u22ef\u2513"))
           (loop [i kd rs (take print-height (vectors a))]
             (when rs
               (let [pad (- (max i 0) first-col)
                     entries (take (- print-width (max 0 pad)) (drop (- pad) (first rs)))]
                 (cl-format w format-band "\u2192"
                            (op (repeat pad pad-str) (map formatter entries)
                                (repeat (- print-width (+ (max 0 pad) (count entries))) pad-str))))
               (recur (dec i) (next rs))))))
       (.write w (if full-height "\u2517" " \u22ee\n\u2517"))))
    ([^java.io.Writer w a]
     (when (< 0 (ecount a))
       (let [max-value (double (amax (engine a) a))
             formatter (partial cl-format nil (if (< max-value 10000.0) format-f format-g))]
         (print-banded w formatter a)))))

  (defn print-packed
    ([^java.io.Writer w formatter ^Matrix a]
     (let [nav (navigator a)]
       (dotimes [j (.ncols a)]
         (print-vector w formatter (.stripe nav a j))
         (.write w "\n"))))
    ([^java.io.Writer w a]
     (let [max-value (double (amax (engine a) a))
           formatter (partial cl-format nil  (if (< max-value 10000.0) format-f format-g))]
       (print-packed w formatter a)))))
