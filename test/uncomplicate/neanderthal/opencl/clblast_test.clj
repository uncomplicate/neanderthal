(ns uncomplicate.neanderthal.opencl.clblast-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core
             :refer [with-default *command-queue*]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.neanderthal
             [opencl :refer [with-engine *opencl-factory* clge]]
             [block-test :as block-test]
             [real-test :as real-test]
             [opencl-test :as opencl-test]]
            [uncomplicate.neanderthal.opencl.clblast
             :refer [clblast-float clblast-double]]))

(with-default
  (with-engine clblast-float *command-queue*
    (block-test/test-all *opencl-factory*)
    (real-test/test-all *opencl-factory*)
    (opencl-test/test-all *opencl-factory*))
  (with-engine clblast-double *command-queue*
    (block-test/test-all *opencl-factory*)
    (real-test/test-all *opencl-factory*)
    (opencl-test/test-all *opencl-factory*)))
