(ns uncomplicate.neanderthal.core
  (:require [primitive-math]
            [uncomplicate.neanderthal.protocols :as p])
  (:import [uncomplicate.neanderthal.protocols
            Vector Matrix Carrier]))

(set! *warn-on-reflection* true)
(primitive-math/use-primitive-operators)

(def ^:private SEGMENT_MSG
  "Required segment %d-%d does not fit in a %d-dim vector.")

(def ^:private ROW_COL_MSG
  "Required %s %d is higher than the row count %d.")

(def ^:private DIMENSION_MSG
  "Different dimensions - %s:%d, %s:%d.")

(defn zero [x]
  (p/zero x))

;; ================= Vector =======================
(defn vect? [x]
  (instance? Vector x))

(defn matrix? [x]
  (instance? Matrix x))

(defn dim ^long [^Vector x]
  (.dim x))

(defn segment [^Vector x ^long k ^long l]
  (.segment x k l))

;; ================= Matrix =======================
(defn mrows ^long [^Matrix m]
  (.mrows m))

(defn ncols ^long [^Matrix m]
  (.ncols m))

(defn row [^Matrix m ^long i]
  (if (< -1 i (.mrows m))
    (.row m i)
    (throw (IndexOutOfBoundsException.
            (format ROW_COL_MSG "row" i (.mrows m))))))

(defn col [^Matrix m ^long j]
  (if (< -1 j (.ncols m))
    (.col m j)
    (throw (IndexOutOfBoundsException.
            (format ROW_COL_MSG "col" j (.ncols m))))))

(defn cols [^Matrix m] ;;TODO
  (map #(.col m %) (range (.ncols m))))

(defn trans [^Matrix m]
  (.transpose m))

;;================== BLAS 1 =======================
(defn iamax ^long [^Vector x]
  (.iamax x))

(defn swapv! [^Vector x ^Vector y]
  (if (= (.dim x) (.dim y))
    (.swap x y)
    (throw (IllegalArgumentException.
            (format DIMENSION_MSG \x (.dim x) \y (.dim y))))))

(defn copy! [^Vector x ^Vector y]
  (if (= (.dim x) (.dim y))
    (.copy x y)
    (throw (IllegalArgumentException.
            (format DIMENSION_MSG \x (.dim x) \y (.dim y))))))

(defn copy [^Carrier x]
  (copy! x (p/zero x)))

(primitive-math/unuse-primitive-operators)
