(ns uncomplicate.neanderthal.core
  (:require [primitive-math]
            [uncomplicate.neanderthal.protocols :as p])
  (:import [uncomplicate.neanderthal.protocols
            Vector Matrix Carrier]))

(set! *warn-on-reflection* true)
(primitive-math/use-primitive-operators)

;; ================= Vector =======================
(defn dim ^long [^Vector x]
  (.dim x))

;; ================= Matrix =======================
(defn mrows ^long [^Matrix m]
  (.mrows m))

(defn ncols ^long [^Matrix m]
  (.ncols m))

(defn row [^Matrix m ^long i]
  (if (< i (.ncols m))
    (.row m)
    (throw (Ille))))
;;================== BLAS 1 =======================
(defn iamax ^long [^Vector x]
  (.iamax x))

(defn swapv! [^Vector x ^Vector y]
  (if (= (.dim x) (.dim y))
    (.swap x y)
    (throw (IllegalArgumentException.
            "Arguments should have the same number of elements."))))

(defn copy! [^Vector x ^Vector y]
  (if (= (.dim x) (.dim y))
    (.copy x y)
    (throw (IllegalArgumentException.
            "Arguments should have the same number of elements."))))

(defn copy [^Carrier x]
  (copy! x (p/pure x)))

(primitive-math/unuse-primitive-operators)
