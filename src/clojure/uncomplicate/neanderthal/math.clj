;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.math
  "Primitive floating point mathematical functions commonly found in Math, FastMath, and the likes.
  Vectorized counterparts can be found in the [[vect-math]] namespace."
  (:import [org.apache.commons.math3.util Precision FastMath]
           [org.apache.commons.math3.special Gamma Erf]))

(defn f=
  ([^double x ^double y ^double nepsilons]
   (Precision/equals x y (* Precision/EPSILON nepsilons)))
  ([^double x ^double y]
   (Precision/equals x y Precision/EPSILON)))

(defn f<
  ([^double x ^double y ^double nepsilons]
   (< (Precision/compareTo x y (* Precision/EPSILON nepsilons)) 0))
  ([^double x ^double y]
   (< (Precision/compareTo x y Precision/EPSILON) 0)))

(defn f<=
  ([^double x ^double y ^double nepsilons]
   (<= (Precision/compareTo x y (* Precision/EPSILON nepsilons)) 0))
  ([^double x ^double y]
   (<= (Precision/compareTo x y Precision/EPSILON) 0)))

(defn f>
  ([^double x ^double y ^double nepsilons]
   (> (Precision/compareTo x y (* Precision/EPSILON nepsilons)) 0))
  ([^double x ^double y]
   (> (Precision/compareTo x y Precision/EPSILON) 0)))

(defn f>=
  ([^double x ^double y ^double nepsilons]
   (>= (Precision/compareTo x y (* Precision/EPSILON nepsilons)) 0))
  ([^double x ^double y]
   (>= (Precision/compareTo x y Precision/EPSILON) 0)))

(defn pow-of-2? [^long n]
  (= 0 (bit-and n (- n 1))))

(defn sqr ^double [^double x]
  (* x x))

(defn abs ^double [^double x]
  (Math/abs x))

(defn sqrt ^double [^double x]
  (Math/sqrt x))

(defn cbrt ^double [^double x]
  (Math/cbrt x))

(defn pow
  (^double [^double x ^double y]
   (Math/pow x y))
  ([^double y]
   (fn ^double [^double x]
     (Math/pow x y))))

(defn hypot ^double [^double x ^double y]
  (Math/hypot x y))

(defn exp ^double [^double x]
  (Math/exp x))

(defn expm1 ^double [^double x]
  (Math/expm1 x))

(defn log ^double [^double x]
  (Math/log x))

(defn log10 ^double [^double x]
  (Math/log10 x))

(defn log1p ^double [^double x]
  (Math/log1p x))

(defn sin ^double [^double x]
  (Math/sin x))

(defn sinh ^double [^double x]
  (Math/sinh x))

(defn asin ^double [^double x]
  (Math/asin x))

(defn asinh ^double [^double x]
  (FastMath/asinh x))

(defn cos ^double [^double x]
  (Math/cos x))

(defn cosh ^double [^double x]
  (Math/cosh x))

(defn acos ^double [^double x]
  (Math/acos x))

(defn acosh ^double [^double x]
  (FastMath/acosh x))

(defn tan ^double [^double x]
  (Math/tan x))

(defn tanh ^double [^double x]
  (Math/tanh x))

(defn atan ^double [^double x]
  (Math/atan x))

(defn atanh ^double [^double x]
  (FastMath/atanh x))

(defn atan2 ^double [^double x ^double y]
  (Math/atan2 x y))

(def ^:const pi Math/PI)

(defn erf
  "Error function: erf(x) = 2/√π 0∫x e-t2dt"
  ^double [^double x]
  (Erf/erf x))

(defn erfc
  "The complementary error function: erfc(x) = 1 - erf(x)"
  ^double [^double x]
  (Erf/erfc x))

(defn erf-inv
  "Inverse error function"
  ^double [^double x]
  (Erf/erfInv x))

(defn erfc-inv
  "Inverse complementary error function"
  ^double [^double x]
  (Erf/erfcInv x))

(let [sqrt2 (sqrt 2.0)]

  (defn cdf-norm
    "The CDF of Normal(0,1)"
    ^double [^double x]
    (* 0.5 (inc (Erf/erf (/ x sqrt2)))))

  (defn cdf-norm-inv
    "The inverse CDF of Normal(0,1)"
    ^double [^double x]
    (* sqrt2 (Erf/erfInv (dec (* 2.0 x))))))

(defn gamma
  "Gamma function: http://en.wikipedia.org/wiki/Gamma_function"
  ^double [^double x]
  (Gamma/gamma x))

(defn lgamma
  "Natural logarithm of the Gamma function:
  http://en.wikipedia.org/wiki/Gamma_function#The_log-gamma_function"
  ^double [^double x]
  (Gamma/logGamma x))

(defn signum ^double [^double x]
  (Math/signum x))

(defn floor ^double [^double x]
  (Math/floor x))

(defn ceil ^double [^double x]
  (Math/ceil x))

(defn trunc ^double [^double x]
  (if (< 0.0 x) (Math/floor x) (Math/ceil x)))

(defn round ^double [^double x]
  (Math/floor (+ 0.5 x)))

(defn frac ^double [^double x]
  (if (< 0.0 x) (- x (Math/floor x)) (- x (Math/ceil x))))

(defn copy-sign ^double [^double x ^double y]
  (Math/copySign x y))

(defn sigmoid ^double [^double x]
  (* 0.5 (inc (Math/tanh (* 0.5 x)))))

(defn ramp ^double [^double x]
  (max 0 x))

(defn relu
  (^double [^double alpha ^double x]
   (max x (* alpha x)))
  ([^double alpha]
   (fn ^double [^double x]
     (max x (* alpha x)))))

(defn elu
  (^double [^double alpha ^double x]
   (max x (* alpha (expm1 x))))
  ([^double alpha]
   (fn ^double [^double x]
     (max x (* alpha (expm1 x))))))

(defn round?
  ([^double x]
   (f= x (Math/floor (+ 0.5 x))))
  ([^double x ^double nepsilons]
   (f= x (Math/floor (+ 0.5 x)) nepsilons)))

(defn magnitude
  (^double [^double range]
   (pow 10.0 (floor (log10 (abs range)))))
  (^double [^double lower ^double upper]
   (magnitude (abs (- upper lower)))))
