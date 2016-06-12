(ns uncomplicate.neanderthal.protocols
  (:import [uncomplicate.neanderthal.protocols Block]))

(defprotocol ReductionFunction
  (vector-reduce [f init x] [f init x y] [f init x y z] [f init x y z v])
  (vector-reduce-map [f init g x] [f init g x y]
    [f init g x y z] [f init g x y z v]))

(defprotocol Factory
  (create-vector [this n source options])
  (create-matrix [this m n source options])
  (vector-engine [this])
  (matrix-engine [this]))

(defprotocol EngineProvider
  (engine ^BLAS [this]))

(defprotocol FactoryProvider
  (factory [this]))

(defprotocol DataAccessorProvider
  (data-accessor ^DataAccessor [this]))

(defprotocol Memory
  (compatible [this other]))

(defprotocol Mappable
  (map-memory [this] [this flags])
  (unmap [this mapped]))

(defprotocol Container
  (raw [this] [this factory])
  (zero [this] [this factory])
  (host [this]))

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

(def ^{:no-doc true :const true} INCOMPATIBLE_BLOCKS_MSG_4
  "Operation is not permited on objects with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  1: %s
  2: %s
  3: %s
  4: %s")

(def ^{:no-doc true :const true} ROW_COL_MSG
  "Required %s %d is higher than the %s count %d.")

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
