(ns uncomplicate.neanderthal.cublas-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [cuda :refer [with-engine *cuda-factory* cuda-handle cuda-float cuda-double]]
             [block-test :as block-test]
             [real-test :as real-test]
             [device-test :as device-test]
             [math-test :as math-test]]))

(defn test-blas-cublas [factory]
  (real-test/test-iamin factory)
  (real-test/test-rot factory)
  (real-test/test-rotm factory))

(with-release [handle (cuda-handle)]

  (with-engine cuda-float handle
    (block-test/test-all *cuda-factory*)
    (real-test/test-blas *cuda-factory*)
    (test-blas-cublas *cuda-factory*)
    (device-test/test-all *cuda-factory*)
    (math-test/test-all-cuda *cuda-factory*))

  (with-engine cuda-double handle
    (block-test/test-all *cuda-factory*)
    (real-test/test-blas *cuda-factory*)
    (test-blas-cublas *cuda-factory*)
    (device-test/test-all *cuda-factory*)
    (math-test/test-all-cuda *cuda-factory*)))
