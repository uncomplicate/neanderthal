(ns uncomplicate.neanderthal.opencl.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core
             :refer [with-default *command-queue*]]
            [uncomplicate.neanderthal
             [opencl :refer [with-gcn-engine *double-factory* *single-factory*]]
             [real-test :as real-test]
             [opencl-test :as opencl-test]]
            [uncomplicate.neanderthal.opencl.amd-gcn :refer [gcn-factory]]))

(with-default
  (with-gcn-engine *command-queue*
    (opencl-test/test-all *single-factory*)
    (opencl-test/test-all *double-factory*)
    (real-test/test-all *single-factory*)
    (real-test/test-all *double-factory*)))
