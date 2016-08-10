(ns uncomplicate.neanderthal.math
  (:import [org.apache.commons.math3.util Precision]))

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

(defn power-of-2? [^long n]
  (= 0 (bit-and n (- n 1))))

(defn round ^double [^double x]
  (Math/floor (+ 0.5 x)))

(defn round?
  ([^double x]
   (f= x (Math/floor (+ 0.5 x))))
  ([^double x ^double nepsilons]
   (f= x (Math/floor (+ 0.5 x)) nepsilons)))

(defn floor ^double [^double x]
  (Math/floor x))

(defn ceil ^double [^double x]
  (Math/ceil x))

(defn pow
  (^double [^double x ^double y]
   (Math/pow x y))
  ([^double y]
   (fn ^double [^double x]
     (Math/pow x y))))

(defn exp ^double [^double x]
  (Math/exp x))

(defn log ^double [^double x]
  (Math/log x))

(defn log10 ^double [^double x]
  (Math/log10 x))

(defn log1p ^double [^double x]
  (Math/log1p x))

(defn sqrt ^double [^double x]
  (Math/sqrt x))

(defn abs ^double [^double x]
  (Math/abs x))

(defn magnitude
  (^double [^double range]
   (pow 10 (floor (log10 range))))
  (^double [^double lower ^double upper]
   (magnitude (- upper lower))))
