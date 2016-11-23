;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.protocols
  (:import [uncomplicate.neanderthal.protocols Block]))

(defprotocol ReductionFunction
  (vector-reduce [f init x] [f init x y] [f init x y z] [f init x y z v])
  (vector-reduce-map [f init g x] [f init g x y]
    [f init g x y z] [f init g x y z v]))

(defprotocol Factory
  (create-vector [this n source options]);;TODO remove options
  (create-matrix [this m n source ord]);;TODO rename to create-ge-matrix
  (create-tr-matrix [this n source ord fuplo fdiag])
  (vector-engine [this])
  (matrix-engine [this])
  (tr-matrix-engine [this]));; rename to ge-matrix-engine

(defprotocol EngineProvider
  (engine ^BLAS [this]))

(defprotocol FactoryProvider
  (factory [this])
  (native-factory [this]))

(defprotocol DataAccessorProvider
  (data-accessor ^DataAccessor [this]))

(defprotocol MemoryContext
  (compatible [this other]))

(defprotocol Mappable
  (map-memory [this] [this flags])
  (unmap [this mapped]))

(defprotocol Container
  (raw [this] [this factory])
  (zero [this] [this factory])
  (host [this])
  (native [this]))

(defprotocol DenseContainer
  (subband [this p q])
  (subtriangle [this uplo diag])
  (is-ge [this]))

(def ^:const ROW_MAJOR 101)
(def ^:const COLUMN_MAJOR 102)
(def ^:const ORDER 103)
(def ^:const DEFAULT_ORDER COLUMN_MAJOR)

(def ^:const UPPER 121)
(def ^:const LOWER 122)
(def ^:const UPLO 123)
(def ^:const DEFAULT_UPLO UPPER)

(def ^:const DIAG_NON_UNIT 131)
(def ^:const DIAG_UNIT 132)
(def ^:const DIAG 135)

(def ^:const LEFT 141)
(def ^:const RIGHT 142)
(def ^:const SIDE 143)

(defn column-major? [^Block a]
  (= COLUMN_MAJOR (.order a)))

(defn dec-property
  [^long code]
  (case code
    101 :row
    102 :column
    103 :order
    111 :no-trans
    112 :trans
    113 :conj-trans
    114 :atlas-conj
    121 :upper
    122 :lower
    123 :uplo
    131 :non-unit
    132 :unit
    135 :diag
    141 :left
    142 :right
    143 :side
    :unknown))

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
