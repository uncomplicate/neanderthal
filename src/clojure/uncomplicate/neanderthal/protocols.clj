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

(defprotocol Container
  (raw [this])
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

(def ^{:no-doc true :const true} MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def ^{:no-doc true :const true} VECTOR_BOUNDS_MSG
  "Requested dim %d is out of bounds of vector dim %d.")

(def ^{:no-doc true :const true} INCOMPATIBLE_BLOCKS_MSG
  "Operation is not permited on objects with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  1: %s
  2: %s")

(def ^{:no-doc true :const true} INCOMPATIBLE_BLOCKS_MSG_3
  "Operation is not permited on objects with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  1: %s
  2: %s
  3: %s")

(def ^{:no-doc true :const true} ROW_COL_MSG
  "Required %s %d is higher than the row count %d.")

(def ^{:no-doc true :const true} DIMENSION_MSG
  "Incompatible dimensions - expected:%d, actual:%d.")

(def ^{:no-doc true :const true} STRIDE_MSG
  "Incompatible stride - expected:%d, actual:%d.")

(def ^{:no-doc true :const true} ILLEGAL_SOURCE_MSG
  "%d is not a valid data source for %s.")

(def ^{:no-doc true :const true} DIMENSIONS_MSG
  "Vector dimensions should be %d.")

(def ^{:no-doc true :const true} PRIMITIVE_FN_MSG
  "I cannot accept function of this type as an argument.")

(def ^{:no-doc true :const true} ROTMG_COND_MSG
  "Arguments p and args must be compatible.
  p must have dimension 5 and args must have dimension 4.
  p: %s;
  args: %s ")

(def ^{:no-doc true :const true} ROTM_COND_MSG
  "Arguments x and y must be compatible and have the same dimensions.
  argument p must have dimension 5.
  x: %s;
  y: %s;
  p: %s")
