(ns uncomplicate.neanderthal.opencl.clblast-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.clojurecl.core
             :refer [with-default *command-queue*]]
            [uncomplicate.fluokitten.core :refer [op]]
            [uncomplicate.neanderthal
             [opencl :refer [with-engine *opencl-factory*]]
             [block-test :as block-test]
             [real-test :as real-test]
             [opencl-test :as opencl-test]]
            [uncomplicate.neanderthal.opencl.clblast
             :refer [clblast-single clblast-double]]))

(defn real-tests [factory]
  (real-test/test-group factory)
  (real-test/test-ge-matrix factory)
  (real-test/test-vector-constructor factory)
  (real-test/test-vector factory)
  (real-test/test-vector-bulk-entry! factory)
  (real-test/test-dot factory)
  (real-test/test-nrm2 factory)
  (real-test/test-asum factory)
  (real-test/test-sum factory)
  (real-test/test-iamax factory)
  (real-test/test-imax factory)
  (real-test/test-imin factory)
  (real-test/test-swap factory)
  (real-test/test-scal factory)
  (real-test/test-vector-copy factory)
  (real-test/test-axpy factory)
  (real-test/test-matrix-constructor factory)
  (real-test/test-matrix factory)
  (real-test/test-matrix-bulk-entry! factory)
  (real-test/test-matrix-copy factory)
  (real-test/test-matrix-scal factory)
  (real-test/test-mv factory)
  (real-test/test-rank factory)
  (real-test/test-mm factory)
  (real-test/test-mm-row-major factory))

(defn block-tests [factory]
  (block-test/test-equality factory)
  (block-test/test-release factory)
  (block-test/test-op-vector factory)
  (block-test/test-op-ge-matrix factory))

(with-default
  (with-engine clblast-single *command-queue*
    (block-tests *opencl-factory*)
    (real-tests *opencl-factory*)
    (opencl-test/test-all *opencl-factory*))
  (with-engine clblast-double *command-queue*
    (block-tests *opencl-factory*)
    (real-tests *opencl-factory*)
    (opencl-test/test-all *opencl-factory*)))
