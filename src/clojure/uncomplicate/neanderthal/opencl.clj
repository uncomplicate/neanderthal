(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.commons.core
             :refer [release let-release wrap-float wrap-double]]
            [uncomplicate.clojurecl
             [core :refer [*context* *command-queue* cl-buffer?]]
             [info :refer [queue-context]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [vect? matrix? transfer!
                           create create-vector create-ge-matrix]]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]
            [uncomplicate.neanderthal.opencl
             [clblock :refer [->TypedCLAccessor cl-to-host host-to-cl]]
             [clblast :refer [clblast-double clblast-single]]])
  (:import [uncomplicate.neanderthal.protocols Block DataAccessor]))

(def ^:dynamic *opencl-factory*)

(def opencl-single clblast-single)

(def opencl-double clblast-double)

(defmacro with-engine
  "Creates a concrete OpenCL factory that executes in the provided queue,
  and binds *opencl-factory* to it. Enables the use of clv and clge in body.

  (with-default
    (with-engine clblast-single *command-queue*
      (with-release [gpu-x (clv (range 3))]
        (sum gpu-x))))
  "
  ([factory queue & body]
   `(binding [*opencl-factory* (~factory (queue-context ~queue) ~queue)]
      (try
        ~@body
        (finally (release *opencl-factory*))))))

(defmacro with-default-engine
  "Creates a concrete single-precision CLBlast OpenCL factory that executes
  in the default context and queue, and binds *opencl-factory* to it.
  clv and clge in its body will create single-precision blocks.

  (with-default
    (with-default-engine
      (with-release [gpu-x (clv (range 3))]
        (sum gpu-x))))
  "
  [& body]
  `(binding [*opencl-factory* (opencl-single *context* *command-queue*)]
     (try
       ~@body
       (finally (release *opencl-factory*)))))

(defn clv
  "Creates an OpenCL-backed vector on the device, with dimension n, using
  the default engine factory.

  (clv 3)
  "
  ([source]
   (create-vector *opencl-factory* source))
  ([x & xs]
   (clv (cons x xs))))

(defn clge
  "Creates an OpenCL-backed matrix on the device, with dimensions m x n,
  using the default engine factory.

  (clge 2 3)
  "
  ([^long m ^long n source]
   (create-ge-matrix *opencl-factory* m n source))
  ([^long m ^long n]
   (create *opencl-factory* m n)))
