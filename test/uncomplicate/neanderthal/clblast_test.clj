(ns uncomplicate.neanderthal.clblast-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core
             :refer [with-default *command-queue*]]
            [uncomplicate.neanderthal
             [core :refer [tr]]
             [opencl :refer [with-engine *opencl-factory*]]
             [block-test :as block-test]
             [real-test :as real-test]
             [device-test :as device-test]
             [math-test :as math-test]]
            [uncomplicate.neanderthal.internal.device.clblast :refer [clblast-float clblast-double]]))

(defn test-blas-clblast [factory]
  (real-test/test-imin factory)
  (real-test/test-imax factory)
  (real-test/test-ge-trans! factory)
  (real-test/test-ge-sum factory))

(defn test-lapack-clblast [factory] ;;TODO This hangs in CLBlast, and I am not sure why!
  (real-test/test-tr-trs factory tr)
  (real-test/test-tr-sv factory tr))

(with-default

  (with-engine clblast-float *command-queue*
    (block-test/test-all *opencl-factory*)
    (real-test/test-blas *opencl-factory*)
    (test-blas-clblast *opencl-factory*)
    ;;(test-lapack-clblast *opencl-factory*)
    (device-test/test-all *opencl-factory*)
    (math-test/test-all-device *opencl-factory*))

  (with-engine clblast-double *command-queue*
    (block-test/test-all *opencl-factory*)
    (real-test/test-blas *opencl-factory*)
    (test-blas-clblast *opencl-factory*)
    ;;(test-lapack-clblast *opencl-factory*)
    (device-test/test-all *opencl-factory*)
    (math-test/test-all-device *opencl-factory*)))
