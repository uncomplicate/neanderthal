(ns uncomplicate.neanderthal.opencl.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core
             :refer [with-default *command-queue*]]
            [uncomplicate.neanderthal
             [opencl :refer [with-gcn-engine *double-factory* *single-factory*]]
             [real-test :as real-test]
             [opencl-test :as opencl-test]]
            [uncomplicate.neanderthal.opencl.amd-gcn :refer [gcn-factory]]))

(defn real-tests [factory]
  (do
    (real-test/test-group factory)
    (real-test/test-ge-matrix factory)
    (real-test/test-vector-constructor factory)
    (real-test/test-vector factory)
    (real-test/test-dot factory)
    (real-test/test-nrm2 factory)
    (real-test/test-asum factory)
    (real-test/test-sum factory)
    (real-test/test-iamax factory)
    (real-test/test-swap factory)
    (real-test/test-scal factory)
    (real-test/test-copy-vector factory)
    (real-test/test-axpy factory)
    (real-test/test-matrix-constructor factory)
    (real-test/test-matrix factory)
    (real-test/test-copy-matrix factory)
    (real-test/test-mv factory)
    ;;(real-test/test-rank factory)
    #_(real-test/test-mm factory)))

(with-default
  (with-gcn-engine *command-queue*
    (opencl-test/test-all *single-factory*)
    (opencl-test/test-all *double-factory*)
    (real-tests *single-factory*)
    (real-tests *double-factory*)))
