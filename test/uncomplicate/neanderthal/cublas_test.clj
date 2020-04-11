(ns uncomplicate.neanderthal.cublas-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecuda.core :refer [with-default default-stream]]
            [uncomplicate.neanderthal
             [core :refer [tr sy]]
             [cuda :refer [with-engine *cuda-factory* cuda-float cuda-double factory-by-type]]
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
         (factory-by-type :int) => (throws ExceptionInfo)
         (factory-by-type :long) => (throws ExceptionInfo))

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
    (random-test/test-all-device *cuda-factory*)))
