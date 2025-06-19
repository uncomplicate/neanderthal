;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.clblast-cpu-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl
             [core :refer [*command-queue* with-platform platforms devices
                           with-queue with-context context command-queue-1]]
             [toolbox :refer [decent-platform]]]
            [uncomplicate.neanderthal
             [core :refer [tr]]
             [opencl :refer [with-engine *opencl-factory* opencl-float opencl-double]]
             [block-test :as block-test]
             [real-test :as real-test]
             [device-test :as device-test]
             [math-test :as math-test]]))

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

      (with-engine opencl-float queue
        (block-test/test-all *opencl-factory*)
        (real-test/test-blas *opencl-factory*)
        (test-blas-clblast *opencl-factory*)
        (test-lapack-clblast *opencl-factory*)
        (device-test/test-all *opencl-factory*)
        (math-test/test-all-device *opencl-factory*))

      (with-engine opencl-double queue
        (block-test/test-all *opencl-factory*)
        (real-test/test-blas *opencl-factory*)
        (test-blas-clblast *opencl-factory*)
        (test-lapack-clblast *opencl-factory*)
        (device-test/test-all *opencl-factory*)
        (math-test/test-all-device *opencl-factory*)))))
