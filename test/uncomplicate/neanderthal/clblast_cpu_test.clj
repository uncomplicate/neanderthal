(ns uncomplicate.neanderthal.clblast-cpu-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl
             [core :refer [*command-queue* with-platform platforms devices with-queue with-context
                           context command-queue-1]]
             [toolbox :refer [decent-platform]]]
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

(defn test-lapack-clblast [factory]
  (real-test/test-tr-trs factory tr)
  (real-test/test-tr-sv factory tr))

(let [devs (some-> (platforms)
                   (decent-platform :cpu)
                   (devices :cpu))]
  (when (< 0 (count devs))
    (with-release [dev (first devs)
                   ctx (context [dev])
                   queue (command-queue-1 ctx dev)]

      (with-engine clblast-float queue
        (block-test/test-all *opencl-factory*)
        (real-test/test-blas *opencl-factory*)
        (test-blas-clblast *opencl-factory*)
        (test-lapack-clblast *opencl-factory*)
        (device-test/test-all *opencl-factory*)
        (math-test/test-all-device *opencl-factory*))

      (with-engine clblast-double queue
        (block-test/test-all *opencl-factory*)
        (real-test/test-blas *opencl-factory*)
        (test-blas-clblast *opencl-factory*)
        (test-lapack-clblast *opencl-factory*)
        (device-test/test-all *opencl-factory*)
        (math-test/test-all-device *opencl-factory*)))))
