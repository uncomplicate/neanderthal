;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.common
  (:require [clojure.pprint :refer [cl-format]]
            [uncomplicate.neanderthal.math :refer [ceil floor]])
  (:import [uncomplicate.neanderthal.internal.api Vector Matrix]))

(def ^:private compile-format #'clojure.pprint/compile-format)

(def format-g (compile-format "~6,2,,1G "))
(def format-f (compile-format "~6,2F "))
(def format-a (compile-format "~3@T~A~3@T"))

(defn format-vector [^java.io.Writer w formatter ^Vector x]
  (let [n-print (min (.dim x) (long *print-length*))
        start-2 (- (.dim x) (floor (/ n-print 2)))]
    (dotimes [i (ceil (/ n-print 2))]
      (cl-format w formatter (.boxedEntry x i)))
    (when (< n-print (.dim x))
      (cl-format w format-a "..."))
    (dotimes [i (floor (/ n-print 2))]
      (cl-format w formatter (.boxedEntry x (+ start-2 i))))))

(defn format-matrix [^java.io.Writer w formatter ^Matrix a max-value]
  (let [pl (long *print-length*)
        m-print (min (.mrows a) pl)
        n-print (min (.ncols a) pl)
        m-start-2 (- (.mrows a) (floor (/ m-print 2)))]
    (dotimes [i (ceil (/ m-print 2))]
      (.write w "\n")
      (format-vector w formatter (.row a i)))
    (when (< m-print (.mrows a))
      (let [width (* 0.1  n-print (.length ^String (cl-format nil formatter max-value)))]
        (dotimes [_ 3]
          (.write w "\n")
          (dotimes [_ width]
            (.write w "     .     ")))
        (.write w "\n")))
    (dotimes [i (floor (/ m-print 2))]
      (.write w "\n")
      (format-vector w formatter (.row a (+ m-start-2 i))))))
