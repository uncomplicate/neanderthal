;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.commons.core
             :refer [release let-release wrap-float wrap-double]]
            [uncomplicate.clojurecl
             [core :refer [*context* *command-queue* cl-buffer?]]
             [info :refer [queue-context]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [vect? matrix? transfer! vctr ge tr]]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-float cblas-double]]
            [uncomplicate.neanderthal.opencl
             [clblock :refer [->TypedCLAccessor cl-to-host host-to-cl]]
             [clblast :refer [clblast-double clblast-float]]])
  (:import [uncomplicate.neanderthal.protocols Block DataAccessor]))

(def ^:dynamic *opencl-factory*)

(def opencl-float clblast-float)

(def opencl-double clblast-double)

(defmacro with-engine
  "Creates a concrete OpenCL factory that executes in the provided queue,
  and binds *opencl-factory* to it. Enables the use of clv and clge in body.

  (with-default
    (with-engine clblast-float *command-queue*
      (with-release [gpu-x (clv (range 3))]
        (sum gpu-x))))
  "
  ([factory queue & body]
   `(binding [*opencl-factory* (~factory (queue-context ~queue) ~queue)]
      (try
        ~@body
        (finally (release *opencl-factory*))))))

(defmacro with-default-engine
  "Creates a concrete float-precision CLBlast OpenCL factory that executes
  in the default context and queue, and binds *opencl-factory* to it.
  clv and clge in its body will create float-precision blocks.

  (with-default
    (with-default-engine
      (with-release [gpu-x (clv (range 3))]
        (sum gpu-x))))
  "
  [& body]
  `(binding [*opencl-factory* (opencl-float *context* *command-queue*)]
     (try
       ~@body
       (finally (release *opencl-factory*)))))

(defn clv
  "Creates an OpenCL-backed vector on the device, with dimension n, using
  the default engine factory.

  (clv 3)
  "
  ([source]
   (vctr *opencl-factory* source))
  ([x & xs]
   (clv (cons x xs))))

(defn clge
  "Creates an OpenCL-backed matrix on the device, with dimensions m x n,
  using the default engine factory.

  (clge 2 3)
  "
  ([^long m ^long n source options]
   (ge *opencl-factory* m n source options))
  ([^long m ^long n arg]
   (ge *opencl-factory* m n arg))
  ([^long m ^long n]
   (ge *opencl-factory* m n))
  ([a]
   (ge *opencl-factory* a)))
