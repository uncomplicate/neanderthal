;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.vect-math
  "Vectorized floating point mathematical functions commonly found in Math, FastMath, and the likes.
  Primitive scalar counterparts can be found in the [[math]] namespace."
  (:require [uncomplicate.commons.core :refer [let-release]]
            [uncomplicate.neanderthal.internal.api :as api]))

(defmacro ^:private defmath!
  ([fname apiname a y]
   `(defn ~fname
      ([~a ~y]
       (if (and (api/fits? ~a ~y) (api/compatible? ~a ~y))
         (~apiname (api/engine ~a) ~a ~y)
         (throw (ex-info "Arguments are not compatible" {:a ~a :y ~y}))))
      ([~a]
       (~apiname (api/engine ~a) ~a ~a))))
  ([fname apiname a b y]
   `(defn ~fname
      ([~a ~b ~y]
       (if (and (api/fits? ~a ~b) (api/compatible? ~a ~b) (api/fits? ~a ~y) (api/compatible? ~a ~y))
         (~apiname (api/engine ~a) ~a ~b ~y)
         (throw (ex-info "Arguments are not compatible" {:a ~a :b ~b :y ~y}))))
      ([~a ~b]
       (if (and (api/fits? ~a ~b) (api/compatible? ~a ~b))
         (~apiname (api/engine ~a) ~a ~b ~a)
         (throw (ex-info "Arguments are not compatible" {:a ~a :b ~b})))))))

(defmacro ^:private defmath
  ([fname apiname a]
   `(defn ~fname [~a]
      (let-release [y# (api/raw ~a)]
        (~apiname (api/engine ~a) ~a y#)
        y#)))
  ([fname apiname a b]
   `(defn ~fname [~a ~b]
      (if (and (api/fits? ~a ~b) (api/compatible? ~a ~b))
        (let-release [y# (api/raw ~a)]
          (~apiname (api/engine ~a) ~a ~b y#)
          y#)
        (throw (ex-info "Arguments are not compatible" {:a ~a :b ~b}))))))

(defmath! sqr! api/sqr a y)
(defmath sqr api/sqr a)

(defmath! mul! api/mul a b y)
(defmath mul api/mul a b)

(defmath! div! api/div a b y)
(defmath div api/div a b)

(defmath! inv! api/inv a y)
(defmath inv api/inv a)

(defmath! abs! api/abs a y)
(defmath abs api/abs a)

(defn linear-frac!
  ([scalea a shifta scaleb b shiftb y]
   (if (and (api/fits? a b) (api/compatible? a b) (api/fits? a y) (api/compatible? a y))
     (api/linear-frac (api/engine a) a b scalea shifta scaleb shiftb y)
     (throw (ex-info "Arguments are not compatible" {:a a :b b :y y}))))
  ([scalea a shifta y]
   (linear-frac! scalea a shifta 0.0 a 1.0 y))
  ([scalea a shifta]
   (linear-frac! scalea a shifta 0.0 a 1.0 a))
  ([a shifta]
   (linear-frac! 1.0 a shifta 0.0 a 1.0 a)))

(defn linear-frac
  ([scalea a shifta scaleb b shiftb]
   (let-release [y (api/raw a)]
     (linear-frac! scalea a shifta scaleb b shiftb y)))
  ([scalea a shifta]
   (linear-frac scalea a shifta 0.0 a 1.0))
  ([a shifta]
   (linear-frac 1.0 a shifta 0.0 a 1.0)))

(defmath! fmod! api/fmod a b y)
(defmath fmod api/fmod a b)

(defmath! frem! api/frem a b y)
(defmath frem api/frem a b)

(defmath! sqrt! api/sqrt a y)
(defmath sqrt api/sqrt a)

(defmath! inv-sqrt! api/inv-sqrt a y)
(defmath inv-sqrt api/inv-sqrt a)

(defmath! cbrt! api/cbrt a y)
(defmath cbrt api/cbrt a)

(defmath! inv-cbrt! api/inv-cbrt a y)
(defmath inv-cbrt api/inv-cbrt a)

(defmath! pow23! api/pow2o3 a y)
(defmath pow23 api/pow2o3 a)

(defmath! pow32! api/pow3o2 a y)
(defmath pow32 api/pow3o2 a)

(defn pow! [a b y]
  (if (and (or (number? b) (and (api/fits? a b) (api/compatible? a b))) (api/fits? a y) (api/compatible? a y))
    (if (number? b)
      (api/powx (api/engine a) a b y)
      (api/pow (api/engine a) a b y))
    (throw (ex-info "Arguments are not compatible" {:a a :b b :y y}))))

(defn pow [a b]
  (let-release [y (api/raw a)]
    (pow! a b y)
    y))

(defmath! hypot! api/hypot a b y)
(defmath hypot api/hypot a b)

(defmath! exp! api/exp a y)
(defmath exp api/exp a)

(defmath! expm1! api/expm1 a y)
(defmath expm1 api/expm1 a)

(defmath! log! api/log a y)
(defmath log api/log a)

(defmath! log10! api/log10 a y)
(defmath log10 api/log10 a)

(defmath! sin! api/sin a y)
(defmath sin api/sin a)

(defmath! cos! api/cos a y)
(defmath cos api/cos a)

(defmath! tan! api/tan a y)
(defmath tan api/tan a)

(defn sincos! [a y z]
  (if (and (api/fits? a y) (api/compatible? a y) (api/fits? a z) (api/compatible? a z))
    (api/sincos (api/engine a) a y z)
    (throw (ex-info "Arguments are not compatible" {:a a :b y :y z}))))

(defn sincos [a]
  (let-release [y (api/raw a)
                z (api/raw a)]
    (sincos! a y z)
    [y z]))

(defmath! asin! api/asin a y)
(defmath asin api/asin a)

(defmath! acos! api/acos a y)
(defmath acos api/acos a)

(defmath! atan! api/atan a y)
(defmath atan api/atan a)

(defmath! atan2! api/atan2 a b y)
(defmath atan2 api/atan2 a b)

(defmath! sinh! api/sinh a y)
(defmath sinh api/sinh a)

(defmath! cosh! api/cosh a y)
(defmath cosh api/cosh a)

(defmath! tanh! api/tanh a y)
(defmath tanh api/tanh a)

(defmath! asinh! api/asinh a y)
(defmath asinh api/asinh a)

(defmath! acosh! api/acosh a y)
(defmath acosh api/acosh a)

(defmath! atanh! api/atanh a y)
(defmath atanh api/atanh a)

(defmath! erf! api/erf a y)
(defmath erf api/erf a)

(defmath! erfc! api/erfc a y)
(defmath erfc api/erfc a)

(defmath! erf-inv! api/erf-inv a y)
(defmath erf-inv api/erf-inv a)

(defmath! erfc-inv! api/erfc-inv a y)
(defmath erfc-inv api/erfc-inv a)

(defmath! cdf-norm! api/cdf-norm a y)
(defmath cdf-norm api/cdf-norm a)

(defmath! cdf-norm-inv! api/cdf-norm-inv a y)
(defmath cdf-norm-inv api/cdf-norm-inv a)

(defmath! gamma! api/gamma a y)
(defmath gamma api/gamma a)

(defmath! lgamma! api/lgamma a y)
(defmath lgamma api/lgamma a)

(defmath! expint1! api/expint1 a y)
(defmath expint1 api/expint1 a)

(defmath! floor! api/floor a y)
(defmath floor api/floor a)

(defmath! ceil! api/fceil a y)
(defmath ceil api/fceil a)

(defmath! trunc! api/trunc a y)
(defmath trunc api/trunc a)

(defmath! round! api/round a y)
(defmath round api/round a)

(defmath! modf! api/modf a y z)

(defn modf [a]
  (let-release [y (api/raw a)
                z (api/raw a)]
    (modf! a y z)
    [y z]))

(defmath! frac! api/frac a y)
(defmath frac api/frac a)

(defmath! fmin! api/fmin a b y)
(defmath fmin api/fmin a b)

(defmath! fmax! api/fmax a b y)
(defmath fmax api/fmax a b)

(defmath! copy-sign! api/copy-sign a b y)
(defmath copy-sign api/copy-sign a b);;TODO

(defmath! sigmoid! api/sigmoid a y)
(defmath sigmoid api/sigmoid a)

(defmath! ramp! api/ramp a y)
(defmath ramp api/ramp a)

(defn relu!
  ([^double alpha a y]
   (if (and (api/fits? a y) (api/compatible? a y))
     (api/relu (api/engine a) alpha a y)
     (throw (ex-info "Arguments are not compatible" {:a a :y y}))))
  ([a y]
   (relu! 0.0 a y))
  ([a]
   (relu! 0.0 a a)))

(defn relu
  ([alpha a]
   (let-release [y (api/raw a)]
     (relu! alpha a y)))
  ([a]
   (relu 0.0 a)))

(defn elu!
  ([^double alpha a y]
   (if (and (api/fits? a y) (api/compatible? a y))
     (api/elu (api/engine a) alpha a y)
     (throw (ex-info "Arguments are not compatible" {:a a :y y}))))
  ([a y]
   (elu! 0.0 a y))
  ([a]
   (elu! 0.0 a a)))

(defn elu
  ([alpha a]
   (let-release [y (api/raw a)]
     (elu! alpha a y)))
  ([a]
   (elu 1.0 a)))
