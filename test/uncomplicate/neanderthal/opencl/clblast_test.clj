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
             :refer [clblast-float clblast-double]]))

(defn real-tests [factory]
  (real-test/test-group factory)
  (real-test/test-vctr-constructor factory)
  (real-test/test-vctr factory)
  (real-test/test-vctr-bulk-entry! factory)
  (real-test/test-dot factory)
  (real-test/test-nrm2 factory)
  (real-test/test-asum factory)
  (real-test/test-sum factory)
  (real-test/test-iamax factory)
  (real-test/test-vctr-swap factory)
  (real-test/test-vctr-copy factory)
  (real-test/test-vctr-scal factory)
  (real-test/test-vctr-axpy factory)
  (real-test/test-ge-constructor factory)
  (real-test/test-ge factory)
  (real-test/test-ge factory)
  (real-test/test-ge-bulk-entry! factory)
  (real-test/test-ge-swap factory)
  (real-test/test-ge-copy factory)
  (real-test/test-ge-scal factory)
  (real-test/test-ge-axpy factory)
  (real-test/test-ge-mv factory)
  (real-test/test-rank factory)
  (real-test/test-ge-mm factory))

(defn block-tests [factory]
  (block-test/test-equality factory)
  (block-test/test-release factory)
  (block-test/test-vctr-op factory)
  (block-test/test-ge-op factory))

(with-default
  (with-engine clblast-float *command-queue*
    (block-tests *opencl-factory*)
    (real-tests *opencl-factory*)
    (opencl-test/test-all *opencl-factory*))
  (with-engine clblast-double *command-queue*
    (block-tests *opencl-factory*)
    (real-tests *opencl-factory*)
    (opencl-test/test-all *opencl-factory*)))
