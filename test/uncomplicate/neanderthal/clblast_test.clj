(ns uncomplicate.neanderthal.clblast-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl
             [core :refer [*command-queue* with-default-1]]]
            [uncomplicate.neanderthal
             [core :refer [tr]]
             [opencl :refer [with-engine *opencl-factory* factory-by-type]]
             [block-test :as block-test]
             [real-test :as real-test]
             [device-test :as device-test]
             [math-test :as math-test]
             [random-test :as random-test]]
            [uncomplicate.neanderthal.internal.device.clblast :refer [clblast-float clblast-double]])
  (:import clojure.lang.ExceptionInfo))

(defn test-blas-clblast [factory]
  (real-test/test-imin factory)
  (real-test/test-imax factory)
  (real-test/test-ge-trans! factory)
  (real-test/test-ge-sum factory))

(defn test-lapack-clblast [factory]
  (real-test/test-tr-trs factory tr)
  (real-test/test-tr-sv factory tr))

(with-default-1

  (facts "factory-by-type test"
         (= clblast-float (factory-by-type :float)) => true
         (= clblast-double (factory-by-type :double)) => true
         (factory-by-type :int) => (throws ExceptionInfo)
         (factory-by-type :long) => (throws ExceptionInfo))

  (with-engine clblast-float *command-queue*
    (block-test/test-all *opencl-factory*)
    (real-test/test-blas *opencl-factory*)
    (test-blas-clblast *opencl-factory*)
    (test-lapack-clblast *opencl-factory*)
    (device-test/test-all *opencl-factory*)
    (math-test/test-all-device *opencl-factory*)
    (random-test/test-all *opencl-factory*)
    (random-test/test-all-device *opencl-factory*))

  (with-engine clblast-double *command-queue*
    (block-test/test-all *opencl-factory*)
    (real-test/test-blas *opencl-factory*)
    (test-blas-clblast *opencl-factory*)
    (test-lapack-clblast *opencl-factory*)
    (device-test/test-all *opencl-factory*)
    (math-test/test-all-device *opencl-factory*)
    (random-test/test-all *opencl-factory*)
    (random-test/test-all-device *opencl-factory*)))
