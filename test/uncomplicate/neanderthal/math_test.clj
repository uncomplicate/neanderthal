;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.math-test
  (:require [midje.sweet :refer [facts throws => roughly truthy just]]
            [uncomplicate.commons.core :refer [release with-release let-release double-fn]]
            [uncomplicate.fluokitten.core :refer [fmap]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge copy axpy! nrm2 scal]]
             [math :as m]
             [vect-math :as vm]])
  (:import clojure.lang.ExceptionInfo))

(defmacro ^:private zero []
  `(roughly 0 0.000001))

(defn diff-vctr-1
  ([factory scalarf vectorf start end by]
   (with-release [a (vctr factory (range start end by))
                  result (vectorf a)
                  expected (fmap scalarf a)]
     (nrm2 (axpy! -1 result expected))))
  ([factory scalarf vectorf]
   (diff-vctr-1 factory scalarf vectorf -1 1 0.07)))

(defn diff-vctr-2
  ([factory scalarf vectorf start end by]
   (with-release [a (vctr factory (range start end by))
                  b (scal 0.9 a)
                  result (vectorf a b)
                  expected (fmap scalarf a b)]
     (nrm2 (axpy! -1 result expected))))
  ([factory scalarf vectorf]
   (diff-vctr-2 factory scalarf vectorf -1 1 0.07))
  ([factory scalarf1 scalarf2 vectorf start end by]
   (with-release [a (vctr factory (range start end by))
                  result (vectorf a)
                  expected1 (fmap scalarf1 a)
                  expected2 (fmap scalarf2 a)]
     [(nrm2 (axpy! -1 (result 0) expected1))
      (nrm2 (axpy! -1 (result 1) expected2))])))

(defn diff-ge-1
  ([factory scalarf vectorf start end by]
   (with-release [a (ge factory 5 3 (range start end by))
                  result (vectorf a)
                  expected (fmap scalarf a)]
     (nrm2 (axpy! -1 result expected))))
  ([factory scalarf vectorf]
   (diff-ge-1 factory scalarf vectorf -1 1 0.13)))

(defn diff-ge-2
  ([factory scalarf vectorf start end by]
   (with-release [a (ge factory 5 3 (range start end by))
                  b (scal 0.9 a)
                  result (vectorf a b)
                  expected (fmap scalarf a b)]
     (nrm2 (axpy! -1 result expected))))
  ([factory scalarf vectorf]
   (diff-ge-2 factory scalarf vectorf -1 1 0.13))
  ([factory scalarf1 scalarf2 vectorf start end by]
   (with-release [a (vctr factory (range start end by))
                  result (vectorf a)
                  expected1 (fmap scalarf1 a)
                  expected2 (fmap scalarf2 a)]
     [(nrm2 (axpy! -1 (result 0) expected1))
      (nrm2 (axpy! -1 (result 1) expected2))])))

;; =========================================================================

(defn test-vctr [factory]
  (facts "vctr-sqr" (diff-vctr-1 factory m/sqr vm/sqr) => (zero))
  (facts "vctr-mul" (diff-vctr-2 factory (double-fn *) vm/mul) => (zero))
  (facts "vctr-div" (diff-vctr-2 factory (double-fn /) vm/div) => (zero))
  (facts "vctr-inv" (diff-vctr-1 factory (double-fn /) vm/inv) => (zero))
  (facts "vctr-abs" (diff-vctr-1 factory m/abs vm/abs) => (zero))
  (facts "vctr-linear-frac"
         (with-release [a (vctr factory 1 2 3)
                        b (vctr factory 2 3 4)
                        expected (vctr factory 5/13 7/17 9/21)
                        result (vm/linear-frac 2 a 3 4 b 5)]
           (nrm2 (axpy! -1 result expected)) => (zero)))
  (facts "vctr-fmod" (diff-vctr-2 factory (double-fn mod) vm/fmod) => (zero))
  (facts "vctr-frem" (diff-vctr-2 factory (double-fn #(rem (double %1) (double %2))) vm/frem) => (zero))
  (facts "vctr-sqrt" (diff-vctr-1 factory m/sqrt vm/sqrt 1 9 0.22) => (zero))
  (facts "vctr-inv-sqrt"
         (diff-vctr-1 factory (double-fn (comp m/sqrt /)) vm/inv-sqrt 1 7 0.23) => (zero))
  (facts "vctr-cbrt" (diff-vctr-1 factory m/cbrt vm/cbrt) => (zero))
  (facts "vctr-inv-cbrt" (diff-vctr-1 factory (double-fn (comp m/cbrt /)) vm/inv-cbrt 1 9 0.32) => (zero))
  (facts "vctr-pow23" (diff-vctr-1 factory (m/pow 2/3) vm/pow23 1 3 0.11) => (zero))
  (facts "vctr-pow32" (diff-vctr-1 factory (m/pow 3/2) vm/pow32 1 5 0.13) => (zero))
  (facts "vctr-pow" (diff-vctr-2 factory m/pow vm/pow 1 3 0.17) => (zero))
  (facts "vctr-powx" (diff-vctr-1 factory (m/pow 2) #(vm/pow % 2)) => (zero))
  (facts "vctr-hypot" (diff-vctr-2 factory m/hypot vm/hypot) => (zero))
  (facts "vctr-exp" (diff-vctr-1 factory m/exp vm/exp) => (zero))
  (facts "vctr-expm1" (diff-vctr-1 factory m/expm1 vm/expm1) => (zero))
  (facts "vctr-log" (diff-vctr-1 factory m/log vm/log 0.1 7 0.33) => (zero))
  (facts "vctr-log10" (diff-vctr-1 factory m/log10 vm/log10 0.1 3 0.07) => (zero))
  (facts "vctr-sin" (diff-vctr-1 factory m/sin vm/sin) => (zero))
  (facts "vctr-cos" (diff-vctr-1 factory m/cos vm/cos) => (zero))
  (facts "vctr-tan" (diff-vctr-1 factory m/tan vm/tan 1 9 0.77) => (zero))
  (facts "vctr-sincos" (diff-vctr-2 factory m/sin m/cos vm/sincos 1 3 0.17) => (just [(zero) (zero)]))
  (facts "vctr-asin" (diff-vctr-1 factory m/asin vm/asin) => (zero))
  (facts "vctr-acos" (diff-vctr-1 factory m/acos vm/acos) => (zero))
  (facts "vctr-atan" (diff-vctr-1 factory m/atan vm/atan 1 9 0.66) => (zero))
  (facts "vctr-atan2" (diff-vctr-2 factory m/atan2 vm/atan2 1 9 0.66) => (zero))
  (facts "vctr-sinh" (diff-vctr-1 factory m/sinh vm/sinh) => (zero))
  (facts "vctr-cosh" (diff-vctr-1 factory m/cosh vm/cosh) => (zero))
  (facts "vctr-tanh" (diff-vctr-1 factory m/tanh vm/tanh 1 9 0.77) => (zero))
  (facts "vctr-asinh" (diff-vctr-1 factory m/asinh vm/asinh) => (zero))
  (facts "vctr-acosh"
         (diff-vctr-1 factory (double-fn (comp m/acosh m/cosh)) (comp vm/acosh vm/cosh)) => (zero))
  (facts "vctr-atanh" (diff-vctr-1 factory m/atanh vm/atanh 0 1 0.06) => (zero))
  (facts "vctr-erf" (diff-vctr-1 factory m/erf vm/erf) => (zero))
  (facts "vctr-erfc" (diff-vctr-1 factory m/erfc vm/erfc) => (zero))
  (facts "vctr-erf-inv"
         (diff-vctr-1 factory (double-fn (comp m/erf-inv m/erf)) (comp vm/erf-inv vm/erf)) => (zero))
  (facts "vctr-erfc-inv"
         (diff-vctr-1 factory (double-fn (comp m/erfc-inv m/erfc)) (comp vm/erfc-inv vm/erfc)) => (zero))
  (facts "vctr-cdf-norm" (diff-vctr-1 factory m/cdf-norm vm/cdf-norm) => (zero))
  (facts "vctr-cdf-norm-inv"
         (diff-vctr-1 factory (double-fn (comp m/cdf-norm-inv m/cdf-norm)) (comp vm/cdf-norm-inv vm/cdf-norm))
         => (zero))
  (facts "vctr-gamma" (diff-vctr-1 factory m/gamma vm/gamma 0.1 7 0.22) => (zero))
  (facts "vctr-lgamma" (diff-vctr-1 factory m/lgamma vm/lgamma 0.1 7 0.22) => (zero))
  (facts "vctr-floor" (diff-vctr-1 factory m/floor vm/floor -5 7 0.22) => (zero))
  (facts "vctr-ceil" (diff-vctr-1 factory m/ceil vm/ceil -5 7 0.342) => (zero))
  (facts "vctr-trunc" (diff-vctr-1 factory m/trunc vm/trunc -5 7 0.342) => (zero))
  (facts "vctr-round" (diff-vctr-1 factory m/round vm/round -5 7 0.342) => (zero))
  (facts "vctr-modf" (diff-vctr-2 factory m/trunc m/frac vm/modf -5 6 0.77) => (just [(zero) (zero)]))
  (facts "vctr-frac" (diff-vctr-1 factory m/frac vm/frac -5 9 0.87) => (zero))
  (facts "vctr-max" (diff-vctr-2 factory (double-fn max) vm/fmax) => (zero))
  (facts "vctr-min" (diff-vctr-2 factory (double-fn min) vm/fmin) => (zero)))

(defn test-ge [factory]
  (facts "ge-sqr" (diff-ge-1 factory m/sqr vm/sqr) => (zero))
  (facts "ge-mul" (diff-ge-2 factory (double-fn *) vm/mul) => (zero))
  (facts "ge-div" (diff-ge-2 factory (double-fn /) vm/div) => (zero))
  (facts "ge-inv" (diff-ge-1 factory (double-fn /) vm/inv) => (zero))
  (facts "ge-abs" (diff-ge-1 factory m/abs vm/abs) => (zero))
  (facts "ge-linear-frac"
         (with-release [a (vctr factory 1 2 3)
                        b (vctr factory 2 3 4)
                        expected (vctr factory 5/13 7/17 9/21)
                        result (vm/linear-frac 2 a 3 4 b 5)]
           (nrm2 (axpy! -1 result expected)) => (zero)))
  (facts "ge-fmod" (diff-ge-2 factory (double-fn mod) vm/fmod) => (zero))
  (facts "ge-frem" (diff-ge-2 factory (double-fn #(rem (double %1) (double %2))) vm/frem) => (zero))
  (facts "ge-sqrt" (diff-ge-1 factory m/sqrt vm/sqrt 1 9 0.22) => (zero))
  (facts "ge-inv-sqrt"
         (diff-ge-1 factory (double-fn (comp m/sqrt /)) vm/inv-sqrt 1 7 0.23) => (zero))
  (facts "ge-cbrt" (diff-ge-1 factory m/cbrt vm/cbrt) => (zero))
  (facts "ge-inv-cbrt" (diff-ge-1 factory (double-fn (comp m/cbrt /)) vm/inv-cbrt 1 9 0.32) => (zero))
  (facts "ge-pow23" (diff-ge-1 factory (m/pow 2/3) vm/pow23 1 3 0.11) => (zero))
  (facts "ge-pow32" (diff-ge-1 factory (m/pow 3/2) vm/pow32 1 5 0.13) => (zero))
  (facts "ge-pow" (diff-ge-2 factory m/pow vm/pow 1 3 0.17) => (zero))
  (facts "ge-powx" (diff-ge-1 factory (m/pow 2) #(vm/pow % 2)) => (zero))
  (facts "ge-hypot" (diff-ge-2 factory m/hypot vm/hypot) => (zero))
  (facts "ge-exp" (diff-ge-1 factory m/exp vm/exp) => (zero))
  (facts "ge-expm1" (diff-ge-1 factory m/expm1 vm/expm1) => (zero))
  (facts "ge-log" (diff-ge-1 factory m/log vm/log 0.1 7 0.33) => (zero))
  (facts "ge-log10" (diff-ge-1 factory m/log10 vm/log10 0.1 3 0.07) => (zero))
  (facts "ge-sin" (diff-ge-1 factory m/sin vm/sin) => (zero))
  (facts "ge-cos" (diff-ge-1 factory m/cos vm/cos) => (zero))
  (facts "ge-tan" (diff-ge-1 factory m/tan vm/tan 1 9 0.77) => (zero))
  (facts "ge-sincos" (diff-ge-2 factory m/sin m/cos vm/sincos 1 3 0.17) => (just [(zero) (zero)]))
  (facts "ge-asin" (diff-ge-1 factory m/asin vm/asin) => (zero))
  (facts "ge-acos" (diff-ge-1 factory m/acos vm/acos) => (zero))
  (facts "ge-atan" (diff-ge-1 factory m/atan vm/atan 1 9 0.66) => (zero))
  (facts "ge-atan2" (diff-ge-2 factory m/atan2 vm/atan2 1 9 0.66) => (zero))
  (facts "ge-sinh" (diff-ge-1 factory m/sinh vm/sinh) => (zero))
  (facts "ge-cosh" (diff-ge-1 factory m/cosh vm/cosh) => (zero))
  (facts "ge-tanh" (diff-ge-1 factory m/tanh vm/tanh 1 9 0.77) => (zero))
  (facts "ge-asinh" (diff-ge-1 factory m/asinh vm/asinh) => (zero))
  (facts "ge-acosh"
         (diff-ge-1 factory (double-fn (comp m/acosh m/cosh)) (comp vm/acosh vm/cosh)) => (zero))
  (facts "ge-atanh" (diff-ge-1 factory m/atanh vm/atanh 0 1 0.06) => (zero))
  (facts "ge-erf" (diff-ge-1 factory m/erf vm/erf) => (zero))
  (facts "ge-erfc" (diff-ge-1 factory m/erfc vm/erfc) => (zero))
  (facts "ge-erf-inv"
         (diff-ge-1 factory (double-fn (comp m/erf-inv m/erf)) (comp vm/erf-inv vm/erf)) => (zero))
  (facts "ge-erfc-inv"
         (diff-ge-1 factory (double-fn (comp m/erfc-inv m/erfc)) (comp vm/erfc-inv vm/erfc)) => (zero))
  (facts "ge-cdf-norm" (diff-ge-1 factory m/cdf-norm vm/cdf-norm) => (zero))
  (facts "ge-cdf-norm-inv"
         (diff-ge-1 factory (double-fn (comp m/cdf-norm-inv m/cdf-norm)) (comp vm/cdf-norm-inv vm/cdf-norm))
         => (zero))
  (facts "ge-gamma" (diff-ge-1 factory m/gamma vm/gamma 0.1 7 0.22) => (zero))
  (facts "ge-lgamma" (diff-ge-1 factory m/lgamma vm/lgamma 0.1 7 0.22) => (zero))
  (facts "ge-floor" (diff-ge-1 factory m/floor vm/floor -5 7 0.22) => (zero))
  (facts "ge-ceil" (diff-ge-1 factory m/ceil vm/ceil -5 7 0.342) => (zero))
  (facts "ge-trunc" (diff-ge-1 factory m/trunc vm/trunc -5 7 0.342) => (zero))
  (facts "ge-round" (diff-ge-1 factory m/round vm/round -5 7 0.342) => (zero))
  (facts "ge-modf" (diff-ge-2 factory m/trunc m/frac vm/modf -5 6 0.77) => (just [(zero) (zero)]))
  (facts "ge-frac" (diff-ge-1 factory m/frac vm/frac -5 9 0.87) => (zero))
  (facts "ge-max" (diff-ge-2 factory (double-fn max) vm/fmax) => (zero))
  (facts "ge-min" (diff-ge-2 factory (double-fn min) vm/fmin) => (zero)))

(defn test-all [factory]
  (test-vctr factory)
  (test-ge factory))
