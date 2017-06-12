;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

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

(defn pow-of-2? [^long n]
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

(defn sqr ^double [^double x]
  (* x x))

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
   (pow 10 (floor (log10 (abs range)))))
  (^double [^double lower ^double upper]
   (magnitude (abs (- upper lower)))))

(defn sin ^double [^double x]
  (Math/sin x))

(defn sinh ^double [^double x]
  (Math/sinh x))

(defn cos ^double [^double x]
  (Math/cos x))

(defn cosh ^double [^double x]
  (Math/cosh x))

(defn tan ^double [^double x]
  (Math/tan x))

(defn tanh ^double [^double x]
  (Math/tanh x))

(def ^:const pi Math/PI)
