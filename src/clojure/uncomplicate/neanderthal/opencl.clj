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
             [amd-gcn :refer [gcn-double gcn-single]]
             [clblast :refer [clblast-double clblast-single]]])
  (:import [uncomplicate.neanderthal.protocols Block DataAccessor]))

(def ^:dynamic *opencl-factory*)

(defmacro with-engine
  ([factory queue & body]
   `(binding [*opencl-factory* (~factory (queue-context ~queue) ~queue)]
      (try
        ~@body
        (finally (release *opencl-factory*))))))

(defmacro with-default-engine
  ([& body]
   `(binding [*opencl-factory* (clblast-single *context* *command-queue*)]
      (try
        ~@body
        (finally (release *opencl-factory*))))))

(defn host [cl]
  (let-release [raw-host (p/raw cl (p/factory (p/factory cl)))]
    (cl-to-host cl raw-host)))

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
