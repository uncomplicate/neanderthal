(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.clojurecl
             [core :refer [*context* *command-queue* cl-buffer? release]]
             [info :refer [queue-context]]]
            [uncomplicate.neanderthal
             [protocols :as p]
             [core :refer [vect? matrix?]]]
            [uncomplicate.neanderthal.opencl
             [clblock :refer [create-vector create-ge-matrix create-buffer
                              double-accessor float-accessor]]
             [amd-gcn :refer [gcn-engine-factory gcn-single gcn-double]]])
  (:import [uncomplicate.neanderthal.protocols Block DataAccessor]))

(def ^:dynamic *double-engine-factory*)
(def ^:dynamic *single-engine-factory*)

(defmacro with-engine
  "Creates the required engine factories for the supported primitive types
  using the provided engine factory creation function and the supplied list of
  params (for most engines, it would contain an OpenCL context and queue), and
  dynamically binds them to the default engine factories for these types.
  Evaluates the body with that binding. Releases the factories
  in the `finally` block.

  The engine factory determines context, device, queue and type of the data.

  Example:

      (with-engine my-engine-factory-constructor [ctx queue]
        (do-opencl-stuff))
  "
  ([engine-factory-fn params & body]
   `(binding [*double-engine-factory*
              (~engine-factory-fn double-accessor ~@params)
              *single-engine-factory*
              (~engine-factory-fn float-accessor ~@params)]
      (try ~@body
           (finally (do (release *double-engine-factory*)
                        (release *single-engine-factory*)))))))

(defmacro with-gcn-engine
  "Creates appropriate engine factory optimized for AMD's GCN devices,
  that produces engines that work in the supplied OpenCL queue, binds it
  and releases after use.
  Evaluates the body with these bindings.
  "
  [queue & body]
  `(with-engine gcn-engine-factory [(queue-context ~queue) ~queue] ~@body))

(defmacro with-default-engine
  "Creates appropriate default engine factory (usually, that's the engine I use
  in development), that produces engines that work in the supplied OpenCL queue,
  binds it and releases after use.
  Evaluates the body with these bindings.
  "
  [& body]
  `(with-engine gcn-engine-factory [*context* *command-queue*] ~@body))

(defn read!
  "Reads the data from the cl container in the device memory to the appropriate
  container in the host memory.
  See ClojureCL's documentation for enq-read!
  "
  [cl host]
  (p/read! cl host))

(defn write!
  "Writes the data from the container in the host memory, to the appropriate
  cl container in the device memory.
  See ClojureCL's documentation for enq-write!
  "
  [cl host]
  (p/write! cl host))

(defn clv
  "Creates an OpenCL-backed vector on the device, with dimension n and an
  optional CL buffer source. If source is not provided, creates a new
  buffer on the device. Uses the supplied engine-factory for the device-specific
  work.

  (clv my-float-engine-factory 100)
  "
  ([engine-factory ^long n source]
   (cond
     (and (vect? source)
          (= (.entryType ^DataAccessor (p/data-accessor engine-factory))
             (.entryType ^Block source)))
     (write! (create-vector engine-factory n
                            (create-buffer (p/data-accessor engine-factory) n))
             source)
     (cl-buffer? source) (create-vector engine-factory n source)
     :default (throw (IllegalArgumentException.
                      (format "I do not know how to create a cl vector from %s"
                              (type source))))))
  ([engine-factory ^long n]
   (create-vector engine-factory n)))

(defn sclv
  "Creates an OpenCL-backed float vector on the device, with dimension n, using
  the default engine factory.

  (sclv 3)
  "
  [^long n]
  (clv *single-engine-factory* n))

(defn dclv
  "Creates an OpenCL-backed double vector on the device, with dimension n, using
  the default engine factory.

  (dclv 3)
  "
  [^long n]
  (clv *double-engine-factory* n))

(defn clge
  "Creates an OpenCL-backed vector on the device, with dimensions m x n and an
  optional CL buffer source. If source is not provided, creates a new
  buffer on the device. Uses the supplied engine-factory for the device-specific
  work.

  (cge my-float-engine-factory 100 33)
  "
  ([engine-factory ^long m ^long n source]
   (cond
     (and (matrix? source)
          (= (.entryType ^DataAccessor (p/data-accessor engine-factory))
             (.entryType ^Block source)))
     (write! (create-ge-matrix
              engine-factory m n
              (create-buffer (p/data-accessor engine-factory) (* m n)))
             source)
     (cl-buffer? source) (create-ge-matrix engine-factory m n source)
     :default (throw (IllegalArgumentException.
                      (format "I do not know how to create a general cl matrix from %s"
                              (type source))))))
  ([engine-factory ^long m^long n]
   (create-ge-matrix engine-factory m n)))

(defn sclge
  "Creates an OpenCL-backed float matrix on the device, with dimensions m x n,
  using the default engine factory.

  (sclge 2 3)
  "
  [^long m ^long n]
  (clge *single-engine-factory* m n))

(defn dclge
  "Creates an OpenCL-backed double matrix on the device, with dimensions m x n,
  using the default engine factory.

  (dclge 2 3)
  "
  [^long m ^long n]
  (clge *double-engine-factory* m n))
