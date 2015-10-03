(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.clojurecl
             [core :refer [*context* *command-queue* cl-buffer? release]]
             [toolbox :refer [wrap-float wrap-double]]
             [info :refer [queue-context]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [vect? matrix? transfer!]]]
            [uncomplicate.neanderthal.opencl
             [clblock :refer [create-vector create-ge-matrix ->TypedCLAccessor]]
             [amd-gcn :refer [gcn-factory]]
             [dummy-engine :refer [dummy-factory]]])
  (:import [uncomplicate.neanderthal.protocols Block DataAccessor]))

(def ^:dynamic *double-factory*)
(def ^:dynamic *single-factory*)

(defn float-accessor [ctx queue]
  (->TypedCLAccessor ctx queue Float/TYPE Float/BYTES float-array wrap-float))

(defn double-accessor [ctx queue]
  (->TypedCLAccessor ctx queue Double/TYPE Double/BYTES double-array wrap-double))

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

(defn clv
  "Creates an OpenCL-backed vector on the device, with dimension n and an
  optional CL buffer source. If source is not provided, creates a new
  buffer on the device. Uses the supplied factory for the device-specific
  work.

  (clv my-float-factory 100)
  "
  ([factory ^long n source]
   (cond
     (and (vect? source)
          (= (.entryType ^DataAccessor (p/data-accessor factory))
             (.entryType ^Block source)))
     (transfer! source
                (create-vector factory n
                               (.createDataSource ^DataAccessor (p/data-accessor factory) n)))
     (cl-buffer? source) (create-vector factory n source)
     :default (throw (IllegalArgumentException.
                      (format "I do not know how to create a cl vector from %s"
                              (type source))))))
  ([factory ^long n]
   (create-vector factory n)))

(defn sclv
  "Creates an OpenCL-backed float vector on the device, with dimension n, using
  the default engine factory.

  (sclv 3)
  "
  [^long n]
  (clv *single-factory* n))

(defn dclv
  "Creates an OpenCL-backed double vector on the device, with dimension n, using
  the default engine factory.

  (dclv 3)
  "
  [^long n]
  (clv *double-factory* n))

(defn clge
  "Creates an OpenCL-backed vector on the device, with dimensions m x n and an
  optional CL buffer source. If source is not provided, creates a new
  buffer on the device. Uses the supplied factory for the device-specific
  work.

  (clge my-float-factory 100 33)
  "
  ([factory ^long m ^long n source]
   (cond
     (and (matrix? source)
          (= (.entryType ^DataAccessor (p/data-accessor factory))
             (.entryType ^Block source)))
     (transfer! source
                (create-ge-matrix
                 factory m n
                 (.createDataSource ^DataAccessor
                                    (p/data-accessor factory) (* m n))))
     (cl-buffer? source) (create-ge-matrix factory m n source)
     :default (throw (IllegalArgumentException.
                      (format "I do not know how to create a general cl matrix from %s"
                              (type source))))))
  ([factory ^long m^long n]
   (create-ge-matrix factory m n)))

(defn sclge
  "Creates an OpenCL-backed float matrix on the device, with dimensions m x n,
  using the default engine factory.

  (sclge 2 3)
  "
  [^long m ^long n]
  (clge *single-factory* m n))

(defn dclge
  "Creates an OpenCL-backed double matrix on the device, with dimensions m x n,
  using the default engine factory.

  (dclge 2 3)
  "
  [^long m ^long n]
  (clge *double-factory* m n))
