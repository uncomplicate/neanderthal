(ns uncomplicate.neanderthal.math
  (:import [org.apache.commons.math3.util FastMath Precision]))

(defn f=
  ([^double x ^double y ^double nepsilons]
     (Precision/equals x y (* Precision/EPSILON nepsilons)))
  ([^double x ^double y]
     (Precision/equals x y Precision/EPSILON)))

(defn f<
  ([^double x ^double y ^double nepsilons]
     (< (Precision/compareTo x y (* Precision/EPSILON nepsilons)) 0))
  ([^double x ^double y]
     (< (Precision/compareTo x y Precision/EPSILON))))

(defn f<=
  ([^double x ^double y ^double nepsilons]
     (<= (Precision/compareTo x y (* Precision/EPSILON nepsilons)) 0))
  ([^double x ^double y]
     (<= (Precision/compareTo x y Precision/EPSILON))))

(defn f>
  ([^double x ^double y ^double nepsilons]
     (> (Precision/compareTo x y (* Precision/EPSILON nepsilons)) 0))
  ([^double x ^double y]
     (> (Precision/compareTo x y Precision/EPSILON))))

(defn f>=
  ([^double x ^double y ^double nepsilons]
     (>= (Precision/compareTo x y (* Precision/EPSILON nepsilons)) 0))
  ([^double x ^double y]
     (>= (Precision/compareTo x y Precision/EPSILON))))

(defn round ^double [^double x]
  (Math/floor (+ 0.5 x)))

(defn round? [^double x]
  (= x (Math/floor (+ 0.5 x))))

(defn floor ^double [^double x]
  (Math/floor x))

(defn pow ^double [^double x ^double y]
  (FastMath/pow x y))

(defn exp ^double [^double x]
  (FastMath/exp x))

(defn log ^double [^double x]
  (Math/log x))

(defn sqrt ^double [^double x]
  (FastMath/sqrt x))

(defn abs ^double [^double x]
  (if (< x 0.0) (- x) x))
