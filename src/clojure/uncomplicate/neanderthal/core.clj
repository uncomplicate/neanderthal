(ns uncomplicate.neanderthal.core
  (:require [primitive-math]
            [uncomplicate.neanderthal.protocols :as p])
  (:import [uncomplicate.neanderthal.protocols
            Vector Matrix Carrier]))

(set! *warn-on-reflection* true)
(primitive-math/use-primitive-operators)

(defn zero [x]
  (p/zero x))

;; ================= Vector =======================
(defn vect? [x]
  (instance? Vector x))

(defn dim ^long [^Vector x]
  (.dim x))

;; ================= Matrix =======================
(defn mrows ^long [^Matrix m]
  (.mrows m))

(defn ncols ^long [^Matrix m]
  (.ncols m))

(defn row [^Matrix m ^long i]
  (if (< i (.mrows m))
    (.row m i)
    (throw (IllegalArgumentException.
            (format "Required row %d is higher than the row count %d."
                    i (.mrows m))))))

(defn col [^Matrix m ^long j]
  (if (< j (.ncols m))
    (.row m j)
    (throw (IllegalArgumentException.
            (format "Required column %d is higher than the column count %d."
                    j (.ncols m))))))


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
  (copy! x (p/zero x)))

(primitive-math/unuse-primitive-operators)
