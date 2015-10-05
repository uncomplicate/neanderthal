(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.clojurecl
             [core :refer [*context* *command-queue* cl-buffer? release]]
             [toolbox :refer [wrap-float wrap-double]]
             [info :refer [queue-context]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [vect? matrix? transfer!
                           create create-vector create-ge-matrix]]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]
            [uncomplicate.neanderthal.opencl
             [clblock :refer [->TypedCLAccessor]]
             [amd-gcn :refer [gcn-factory]]
             [dummy-engine :refer [dummy-factory]]])
  (:import [uncomplicate.neanderthal.protocols Block DataAccessor]))

(def ^:dynamic *double-factory*)
(def ^:dynamic *single-factory*)

(defn float-accessor [ctx queue]
  (->TypedCLAccessor ctx queue Float/TYPE Float/BYTES
                     float-array wrap-float cblas-single))

(defn double-accessor [ctx queue]
  (->TypedCLAccessor ctx queue Double/TYPE Double/BYTES
                     double-array wrap-double cblas-double))

(defn gcn-single [ctx queue]
  (gcn-factory float-accessor ctx queue))

(defn gcn-double [ctx queue]
  (gcn-factory double-accessor ctx queue))

(defn dummy-single [ctx queue]
  (dummy-factory float-accessor ctx queue))

(defn dummy-double [ctx queue]
  (dummy-factory double-accessor ctx queue))

(defmacro with-engine
  "Creates the required engine factories for the supported primitive types
  using the provided engine factory creation function and the supplied list of
  params (for most engines, it would contain an OpenCL context and queue), and
  dynamically binds them to the default engine factories for these types.
  Evaluates the body with that binding. Releases the factories
  in the `finally` block.

  The engine factory determines context, device, queue and type of the data.

  Example:

      (with-engine my-factory-constructor [ctx queue]
        (do-opencl-stuff))
  "
  ([factory-fn params & body]
   `(binding [*double-factory*
              (~factory-fn double-accessor ~@params)
              *single-factory*
              (~factory-fn float-accessor ~@params)]
      (try ~@body
           (finally (and (release *double-factory*)
                         (release *single-factory*)))))))

(defmacro with-gcn-engine
  "Creates appropriate engine factory optimized for AMD's GCN devices,
  that produces engines that work in the supplied OpenCL queue, binds it
  and releases after use.
  Evaluates the body with these bindings.
  "
  [queue & body]
  `(with-engine gcn-factory [(queue-context ~queue) ~queue] ~@body))

(defmacro with-default-engine
  "Creates appropriate default engine factory (usually, that's the engine I use
  in development), that produces engines that work in the supplied OpenCL queue,
  binds it and releases after use.
  Evaluates the body with these bindings.
  "
  [& body]
  `(with-engine gcn-factory [*context* *command-queue*] ~@body))

(defn sv-cl
  "Creates an OpenCL-backed float vector on the device, with dimension n, using
  the default engine factory.

  (sv-cl 3)
  "
  ([source]
   (create-vector *single-factory* source))
  ([x & xs]
   (sv-cl (cons x xs))))

(defn dv-cl
  "Creates an OpenCL-backed double vector on the device, with dimension n, using
  the default engine factory.

  (dclv 3)
  "
  ([source]
   (create-vector *double-factory* source))
  ([x & xs]
   (dv-cl (cons x xs))))

(defn sge-cl
  "Creates an OpenCL-backed float matrix on the device, with dimensions m x n,
  using the default engine factory.

  (sge-cl 2 3)
  "
  ([^long m ^long n source]
   (create-ge-matrix *single-factory* m n source))
  ([^long m ^long n]
   (create *single-factory* m n)))

(defn dge-cl
  "Creates an OpenCL-backed double matrix on the device, with dimensions m x n,
  using the default engine factory.

  (dge-cl 2 3)
  "
  ([^long m ^long n source]
   (create-ge-matrix *double-factory* m n source))
  ([^long m ^long n]
   (create *double-factory* m n)))
