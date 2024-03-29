;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.cublas-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecuda.core :refer [with-default default-stream]]
            [uncomplicate.neanderthal
             [core :refer [tr sy]]
             [cuda :refer [with-engine *cuda-factory*  factory-by-type cuda-float
                           cuda-double cuda-long cuda-int cuda-short cuda-byte]]
             [block-test :as block-test]
             [real-test :as real-test]
             [device-test :as device-test]
             [math-test :as math-test]
             [random-test :as random-test]])
  (:import clojure.lang.ExceptionInfo))

(defn test-blas-cublas [factory]
  (real-test/test-iamin factory)
  (real-test/test-rot factory)
  (real-test/test-rotm factory)
  (real-test/test-ge-sum factory)
  (real-test/test-tr-sum factory tr)
  (real-test/test-sy-sum factory sy))

(defn test-lapack-cublas [factory]
  (real-test/test-tr-sv factory tr)
  (real-test/test-tr-trs factory tr))

(defn test-math-cuda [factory]
  (math-test/test-math-inv factory math-test/diff-vctr-1 math-test/diff-vctr-2)
  (math-test/test-math-inv factory math-test/diff-ge-1 math-test/diff-ge-2)
  (math-test/test-math-inv factory (partial math-test/diff-square-1 tr)
                           (partial math-test/diff-square-2 tr)))

(with-default

  (facts "factory-by-type test"
         (= cuda-float (factory-by-type :float)) => true
         (= cuda-double (factory-by-type :double)) => true
         (= cuda-int (factory-by-type :int)) => true
         (= cuda-long (factory-by-type :long)) => true
         (= cuda-short (factory-by-type :short)) => true
         (= cuda-byte (factory-by-type :byte)) => true
         (= cuda-byte (factory-by-type :uint8)) => true)

  (with-engine cuda-float default-stream
    (block-test/test-all *cuda-factory*)
    (real-test/test-blas *cuda-factory*)
    (test-blas-cublas *cuda-factory*)
    (test-lapack-cublas *cuda-factory*)
    (device-test/test-all *cuda-factory*)
    (math-test/test-all-device *cuda-factory*)
    (test-math-cuda *cuda-factory*)
    (random-test/test-all *cuda-factory*)
    (random-test/test-all-device *cuda-factory*))

  (with-engine cuda-double default-stream
    (block-test/test-all *cuda-factory*)
    (real-test/test-blas *cuda-factory*)
    (test-blas-cublas *cuda-factory*)
    (test-lapack-cublas *cuda-factory*)
    (device-test/test-all *cuda-factory*)
    (math-test/test-all-device *cuda-factory*)
    (test-math-cuda *cuda-factory*)
    (random-test/test-all *cuda-factory*)
    (random-test/test-all-device *cuda-factory*))

  (with-engine cuda-long default-stream
    (real-test/test-basic-integer *cuda-factory*))

  (with-engine cuda-int default-stream
    (real-test/test-basic-integer *cuda-factory*))

  (with-engine cuda-short default-stream
    (real-test/test-basic-integer *cuda-factory*))

  (with-engine cuda-byte default-stream
    (real-test/test-basic-integer *cuda-factory*)))
