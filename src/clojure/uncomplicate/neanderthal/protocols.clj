(ns uncomplicate.neanderthal.protocols
  (:import [uncomplicate.neanderthal.protocols Block]))

(defprotocol Factory
  (create-vector [this n source options])
  (create-matrix [this m n source options])
  (data-accessor ^DataAccessor [this])
  (vector-engine [this options])
  (matrix-engine [this options]))

(defprotocol EngineProvider
  (engine ^BLAS [this]))

(defprotocol FactoryProvider
  (factory [this]))

(defprotocol Memory
  (compatible [this other]))

(defprotocol Mappable
  (map-memory [this] [this flags])
  (unmap [this mapped]))

(defprotocol Group
  (zero [this]))

(defprotocol Functor
  (fmap! [x f] [x f y] [x f y z]
    [x f y z v] [x f y z v ws]))

(defprotocol Foldable
  (fold [x] [x f] [x f acc]))

(defprotocol Reducible
  (freduce [x f] [x acc f] [x acc f y]
    [x acc f y z] [x acc f y z ws]))

(def ^:const ROW_MAJOR 101)

(def ^:const COLUMN_MAJOR 102)

(def ^:const DEFAULT_ORDER COLUMN_MAJOR)

(defn column-major? [^Block a]
  (= COLUMN_MAJOR (.order a)))
