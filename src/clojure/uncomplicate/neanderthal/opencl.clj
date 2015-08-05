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
  ([engine-factory-fn params & body]
   `(binding [*double-engine-factory*
              (~engine-factory-fn double-accessor ~@params)
              *single-engine-factory*
              (~engine-factory-fn float-accessor ~@params)]
      (try ~@body
           (finally (do (release *double-engine-factory*)
                        (release *single-engine-factory*)))))))

(defmacro with-gcn-engine [queue & body]
  `(with-engine gcn-engine-factory [(queue-context ~queue) ~queue] ~@body))

(defmacro with-default-engine [& body]
  `(with-engine gcn-engine-factory [*context* *command-queue*] ~@body))

(defn read! [cl host]
  (p/read! cl host))

(defn write! [cl host]
  (p/write! cl host))

(defn clv
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

(defn sclv [^long n]
  (clv *single-engine-factory* n))

(defn dclv [^long n]
  (clv *double-engine-factory* n))

(defn clge
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

(defn sclge [^long m ^long n]
  (clge *single-engine-factory* m n))

(defn dclge [^long m ^long n]
  (clge *double-engine-factory* m n))
