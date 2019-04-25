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
             [core :refer [vctr ge sy tr tp sp gd gt dt st copy axpy! nrm2 scal transfer]]
             [math :as m]
             [vect-math :as vm]]
            [uncomplicate.neanderthal.internal.api :refer [native-factory]])
  (:import clojure.lang.ExceptionInfo))

(defmacro ^:private zero []
  `(roughly 0 0.00001))

(defn diff-vctr-1
  ([factory scalarf vectorf start end by]
   (with-release [a (vctr factory (range start end by))
                  result (vectorf a)
                  host-result (transfer result)
                  host-a (transfer a)
                  expected (fmap scalarf host-a)]
     (nrm2 (axpy! -1 host-result expected))))
  ([factory scalarf vectorf]
   (diff-vctr-1 factory scalarf vectorf -1 1 0.07)))

(defn diff-vctr-2
  ([factory scalarf vectorf start end by]
   (with-release [a (vctr factory (range start end by))
                  b (scal 0.9 a)
                  result (vectorf a b)
                  host-result (transfer result)
                  host-a (transfer a)
                  host-b (transfer b)
                  expected (fmap scalarf host-a host-b)]
     (nrm2 (axpy! -1 host-result expected))))
  ([factory scalarf vectorf]
   (diff-vctr-2 factory scalarf vectorf -1 1 0.07))
  ([factory scalarf1 scalarf2 vectorf start end by]
   (with-release [a (vctr factory (range start end by))
                  result (vectorf a)
                  host-result-1 (transfer (result 0))
                  host-result-2 (transfer (result 1))
                  host-a (transfer a)
                  expected1 (fmap scalarf1 host-a)
                  expected2 (fmap scalarf2 host-a)]
     [(nrm2 (axpy! -1 host-result-1 expected1))
      (nrm2 (axpy! -1 host-result-2 expected2))])))

(defn diff-ge-1
  ([factory scalarf vectorf start end by]
   (with-release [a (ge factory 5 3 (range start end by))
                  result (vectorf a)
                  host-result (transfer result)
                  host-a (transfer a)
                  expected (fmap scalarf host-a)]
     (nrm2 (axpy! -1 host-result expected))))
  ([factory scalarf vectorf]
   (diff-ge-1 factory scalarf vectorf -1 1 0.13)))

(defn diff-ge-2
  ([factory scalarf vectorf start end by]
   (with-release [a (ge factory 5 3 (range start end by))
                  b (scal 0.9 a)
                  result (vectorf a b)
                  host-result (transfer result)
                  host-a (transfer a)
                  host-b (transfer b)
                  expected (fmap scalarf host-a host-b)]
     (nrm2 (axpy! -1 host-result expected))))
  ([factory scalarf vectorf]
   (diff-ge-2 factory scalarf vectorf -1 1 0.13))
  ([factory scalarf1 scalarf2 vectorf start end by]
   (with-release [a (ge factory 5 3(range start end by))
                  result (vectorf a)
                  host-result-1 (transfer (result 0))
                  host-result-2 (transfer (result 1))
                  host-a (transfer a)
                  expected1 (fmap scalarf1 host-a)
                  expected2 (fmap scalarf2 host-a)]
     [(nrm2 (axpy! -1 host-result-1 expected1))
      (nrm2 (axpy! -1 host-result-2 expected2))])))

(defn diff-square-1
  ([create factory scalarf vectorf start end by]
   (with-release [a (create factory 4 (range start end by))
                  result (vectorf a)
                  host-result (transfer result)
                  host-a (transfer a)
                  expected (fmap scalarf host-a)]
     (nrm2 (axpy! -1 host-result expected))))
  ([create factory scalarf vectorf]
   (diff-square-1 create factory scalarf vectorf -1 1 0.13)))

(defn diff-square-2
  ([create factory scalarf vectorf start end by]
   (with-release [a (create factory 4 (range start end by))
                  b (scal 0.9 a)
                  result (vectorf a b)
                  host-result (transfer result)
                  host-a (transfer a)
                  host-b (transfer b)
                  expected (fmap scalarf host-a host-b)]
     (nrm2 (axpy! -1 host-result expected))))
  ([create factory scalarf vectorf]
   (diff-square-2 create factory scalarf vectorf -1 1 0.13))
  ([create factory scalarf1 scalarf2 vectorf start end by]
   (with-release [a (create factory 4 (range start end by))
                  result (vectorf a)
                  host-result-1 (transfer (result 0))
                  host-result-2 (transfer (result 1))
                  host-a (transfer a)
                  expected1 (fmap scalarf1 host-a)
                  expected2 (fmap scalarf2 host-a)]
     [(nrm2 (axpy! -1 host-result-1 expected1))
      (nrm2 (axpy! -1 host-result-2 expected2))])))

;; =========================================================================

(defn test-math [factory diff-1 diff-2]
  (facts "sqr" (diff-1 factory m/sqr vm/sqr) => (zero))
  (facts "mul" (diff-2 factory (double-fn *) vm/mul) => (zero))
  (facts "div" (diff-2 factory (double-fn /) vm/div) => (zero))
  (facts "inv" (diff-1 factory (double-fn /) vm/inv) => (zero))
  (facts "abs" (diff-1 factory m/abs vm/abs) => (zero))
  (facts "fmod" (diff-2 factory (double-fn mod) vm/fmod) => (zero))
  (facts "frem" (diff-2 factory (double-fn #(rem (double %1) (double %2))) vm/frem) => (zero))
  (facts "sqrt" (diff-1 factory m/sqrt vm/sqrt 1 9 0.22) => (zero))
  (facts "inv-sqrt"
         (diff-1 factory (double-fn (comp m/sqrt /)) vm/inv-sqrt 1 7 0.23) => (zero))
  (facts "cbrt" (diff-1 factory m/cbrt vm/cbrt) => (zero))
  (facts "inv-cbrt" (diff-1 factory (double-fn (comp m/cbrt /)) vm/inv-cbrt 1 9 0.32) => (zero))
  (facts "pow23" (diff-1 factory (m/pow 2/3) vm/pow23 1 3 0.11) => (zero))
  (facts "pow32" (diff-1 factory (m/pow 3/2) vm/pow32 1 5 0.13) => (zero))
  (facts "pow" (diff-2 factory m/pow vm/pow 1 3 0.17) => (zero))
  (facts "powx" (diff-1 factory (m/pow 2) #(vm/pow % 2)) => (zero))
  (facts "hypot" (diff-2 factory m/hypot vm/hypot) => (zero))
  (facts "exp" (diff-1 factory m/exp vm/exp) => (zero))
  (facts "expm1" (diff-1 factory m/expm1 vm/expm1) => (zero))
  (facts "log" (diff-1 factory m/log vm/log 0.1 7 0.33) => (zero))
  (facts "log10" (diff-1 factory m/log10 vm/log10 0.1 3 0.07) => (zero))
  (facts "sin" (diff-1 factory m/sin vm/sin) => (zero))
  (facts "cos" (diff-1 factory m/cos vm/cos) => (zero))
  (facts "tan" (diff-1 factory m/tan vm/tan 1 9 0.77) => (zero))
  (facts "sincos" (diff-2 factory m/sin m/cos vm/sincos 1 3 0.17) => (just [(zero) (zero)]))
  (facts "asin" (diff-1 factory m/asin vm/asin) => (zero))
  (facts "acos" (diff-1 factory m/acos vm/acos) => (zero))
  (facts "atan" (diff-1 factory m/atan vm/atan 1 9 0.66) => (zero))
  (facts "atan2" (diff-2 factory m/atan2 vm/atan2 1 9 0.66) => (zero))
  (facts "sinh" (diff-1 factory m/sinh vm/sinh) => (zero))
  (facts "cosh" (diff-1 factory m/cosh vm/cosh) => (zero))
  (facts "tanh" (diff-1 factory m/tanh vm/tanh 1 9 0.77) => (zero))
  (facts "asinh" (diff-1 factory m/asinh vm/asinh) => (zero))
  (facts "acosh"
         (diff-1 factory (double-fn (comp m/acosh m/cosh)) (comp vm/acosh vm/cosh)) => (zero))
  (facts "atanh" (diff-1 factory m/atanh vm/atanh 0 1 0.06) => (zero))
  (facts "erf" (diff-1 factory m/erf vm/erf) => (zero))
  (facts "erfc" (diff-1 factory m/erfc vm/erfc) => (zero))
  (facts "cdf-norm" (diff-1 factory m/cdf-norm vm/cdf-norm) => (zero))
  (facts "lgamma" (diff-1 factory m/lgamma vm/lgamma 0.1 7 0.22) => (zero))
  (facts "floor" (diff-1 factory m/floor vm/floor -5 7 0.22) => (zero))
  (facts "ceil" (diff-1 factory m/ceil vm/ceil -5 7 0.342) => (zero))
  (facts "trunc" (diff-1 factory m/trunc vm/trunc -5 7 0.342) => (zero))
  (facts "round" (diff-1 factory m/round vm/round -5 7 0.342) => (zero))
  (facts "modf" (diff-2 factory m/trunc m/frac vm/modf -5 6 0.77) => (just [(zero) (zero)]))
  (facts "frac" (diff-1 factory m/frac vm/frac -5 9 0.87) => (zero))
  (facts "max" (diff-2 factory (double-fn max) vm/fmax) => (zero))
  (facts "min" (diff-2 factory (double-fn min) vm/fmin) => (zero))
  (facts "copy-sign" (diff-2 factory m/copy-sign vm/copy-sign) => (zero))
  (facts "sigmoid" (diff-1 factory m/sigmoid vm/sigmoid) => (zero))
  (facts "ramp" (diff-1 factory m/ramp vm/ramp) => (zero))
  (facts "relu" (diff-1 factory (m/relu 0.01) #(vm/relu 0.01 %)) => (zero))
  (facts "elu" (diff-1 factory (m/elu 0.01) #(vm/elu 0.01 %)) => (zero)))

(defn test-math-inv [factory diff-1 diff-2]
  (facts "erf-inv"
         (diff-1 factory (double-fn (comp m/erf-inv m/erf)) (comp vm/erf-inv vm/erf)) => (zero))
  (facts "erfc-inv"
         (diff-1 factory (double-fn (comp m/erfc-inv m/erfc)) (comp vm/erfc-inv vm/erfc)) => (zero))
  (facts "cdf-norm-inv"
         (diff-1 factory (double-fn (comp m/cdf-norm-inv m/cdf-norm)) (comp vm/cdf-norm-inv vm/cdf-norm))
         => (zero)))

(defn test-math-host [factory diff-1 diff-2]
  (facts "gamma" (diff-1 factory m/gamma vm/gamma 0.1 7 0.22) => (zero)))

(defn test-math-device [factory diff-1 diff-2]
  (facts "gamma" (diff-1 factory m/gamma vm/gamma 0.1 7 0.22) => (roughly 0 0.001)))

(defn test-vctr-linear-frac [factory]
  (facts "vctr-linear-frac"
         (with-release [a (vctr factory 1 2 3)
                        b (vctr factory 2 3 4)
                        expected (vctr factory 5/13 7/17 9/21)
                        result (vm/linear-frac 2 a 3 4 b 5)]
           (nrm2 (axpy! -1 result expected)) => (zero))))

(defn test-ge-linear-frac [factory]
  (facts "ge-linear-frac"
         (with-release [a (ge factory 3 2 [1 2 3 1 2 3])
                        b (ge factory 3 2 [2 3 4 2 3 4])
                        host-expected (ge (native-factory factory) 3 2 [5/13 7/17 9/21 5/13 7/17 9/21])
                        result (vm/linear-frac 2 a 3 4 b 5)
                        host-result (transfer result)]
           (nrm2 (axpy! -1 host-result host-expected)) => (zero))))

(defn test-tr-linear-frac [factory]
  (facts "tr-linear-frac"
         (with-release [a (tr factory 3 [1 2 3 1 2 3])
                        b (tr factory 3 [2 3 4 2 3 4])
                        host-expected (tr (native-factory factory) 3 [5/13 7/17 9/21 5/13 7/17 9/21])
                        result (vm/linear-frac 2 a 3 4 b 5)
                        host-result (transfer result)]
           (nrm2 (axpy! -1 host-result host-expected)) => (zero))))

(defn test-sy-linear-frac [factory]
  (facts "sy-linear-frac"
         (with-release [a (sy factory 3 [1 2 3 1 2 3])
                        b (sy factory 3 [2 3 4 2 3 4])
                        host-expected (sy (native-factory factory) 3 [5/13 7/17 9/21 5/13 7/17 9/21])
                        result (vm/linear-frac 2 a 3 4 b 5)
                        host-result (transfer result)]
           (nrm2 (axpy! -1 host-result host-expected)) => (zero))))

(defn test-all-host [factory]
  (test-math factory diff-vctr-1 diff-vctr-2)
  (test-math factory diff-ge-1 diff-ge-2)
  (test-math factory (partial diff-square-1 tr) (partial diff-square-2 tr))
  (test-math factory (partial diff-square-1 sy) (partial diff-square-2 sy))
  (test-math factory (partial diff-square-1 tp) (partial diff-square-2 tp))
  (test-math factory (partial diff-square-1 sp) (partial diff-square-2 sp))
  (test-math factory (partial diff-square-1 gd) (partial diff-square-2 gd))
  (test-math factory (partial diff-square-1 gt) (partial diff-square-2 gt))
  (test-math factory (partial diff-square-1 dt) (partial diff-square-2 dt))
  (test-math factory (partial diff-square-1 st) (partial diff-square-2 st))
  (test-math-host factory diff-vctr-1 diff-vctr-2)
  (test-math-host factory diff-ge-1 diff-ge-2)
  (test-math-host factory (partial diff-square-1 tr) (partial diff-square-2 tr))
  (test-math-host factory (partial diff-square-1 sy) (partial diff-square-2 sy))
  (test-math-host factory (partial diff-square-1 tp) (partial diff-square-2 tp))
  (test-math-host factory (partial diff-square-1 sp) (partial diff-square-2 sp))
  (test-math-host factory (partial diff-square-1 gd) (partial diff-square-2 gd))
  (test-math-host factory (partial diff-square-1 gt) (partial diff-square-2 gt))
  (test-math-host factory (partial diff-square-1 dt) (partial diff-square-2 dt))
  (test-math-host factory (partial diff-square-1 st) (partial diff-square-2 st))
  (test-math-inv factory diff-vctr-1 diff-vctr-2)
  (test-math-inv factory diff-ge-1 diff-ge-2)
  (test-math-inv factory (partial diff-square-1 tr) (partial diff-square-2 tr))
  (test-math-inv factory (partial diff-square-1 sy) (partial diff-square-2 sy))
  (test-math-inv factory (partial diff-square-1 tp) (partial diff-square-2 tp))
  (test-math-inv factory (partial diff-square-1 sp) (partial diff-square-2 sp))
  (test-math-inv factory (partial diff-square-1 gd) (partial diff-square-2 gd))
  (test-math-inv factory (partial diff-square-1 gt) (partial diff-square-2 gt))
  (test-math-inv factory (partial diff-square-1 dt) (partial diff-square-2 dt))
  (test-math-inv factory (partial diff-square-1 st) (partial diff-square-2 st))
  (test-vctr-linear-frac factory)
  (test-ge-linear-frac factory)
  (test-tr-linear-frac factory)
  (test-sy-linear-frac factory))

(defn test-all-device [factory]
  (test-math factory diff-vctr-1 diff-vctr-2)
  (test-math factory diff-ge-1 diff-ge-2)
  (test-math factory (partial diff-square-1 tr) (partial diff-square-2 tr))
  (test-math-device factory diff-vctr-1 diff-vctr-2)
  (test-math-device factory diff-ge-1 diff-ge-2)
  (test-math-device factory (partial diff-square-1 tr) (partial diff-square-2 tr))
  (test-vctr-linear-frac factory)
  (test-ge-linear-frac factory)
  (test-tr-linear-frac factory))
