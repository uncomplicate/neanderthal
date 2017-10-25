(ns uncomplicate.neanderthal.cublas-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [tr]]
             [cuda :refer [with-engine *cuda-factory* cuda-handle cuda-float cuda-double]]
             [block-test :as block-test]
             [real-test :as real-test]
             [device-test :as device-test]
             [math-test :as math-test]]))

(defn test-blas-cublas [factory]
  (real-test/test-iamin factory)
  (real-test/test-rot factory)
  (real-test/test-rotm factory))

(defn test-lapack-cublas [factory]
  (real-test/test-tr-sv factory tr)
  (real-test/test-tr-trs factory tr))

(defn test-math-cuda [factory]
  (math-test/test-math-inv factory math-test/diff-vctr-1 math-test/diff-vctr-2)
  (math-test/test-math-inv factory math-test/diff-ge-1 math-test/diff-ge-2)
  (math-test/test-math-inv factory (partial math-test/diff-square-1 tr) (partial math-test/diff-square-2 tr)))

(with-release [handle (cuda-handle)]

  (with-engine cuda-float handle
    (block-test/test-all *cuda-factory*)
    (real-test/test-blas *cuda-factory*)
    (test-blas-cublas *cuda-factory*)
    (test-lapack-cublas *cuda-factory*)
    (device-test/test-all *cuda-factory*)
    (math-test/test-all-device *cuda-factory*)
    (test-math-cuda *cuda-factory*))

  (with-engine cuda-double handle
    (block-test/test-all *cuda-factory*)
    (real-test/test-blas *cuda-factory*)
    (test-blas-cublas *cuda-factory*)
    (test-lapack-cublas *cuda-factory*)
    (device-test/test-all *cuda-factory*)
    (math-test/test-all-device *cuda-factory*)
    (test-math-cuda *cuda-factory*)))
