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
  (create-vector [this memory n])
  (create-ge [this memory m n ord])
  (create-tr [this memory n ord uplo diag])
  (vector-engine [this])
  (ge-engine [this])
  (tr-engine [this]))

(defprotocol EngineProvider
  (engine ^BLAS [this]))

(defprotocol FactoryProvider
  (factory [this])
  (native-factory [this]))

(defprotocol DataAccessorProvider
  (data-accessor ^DataAccessor [this]))

(defprotocol MemoryContext
  (compatible? [this other])
  (fits? [this other]))

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
  (subtriangle [this uplo diag]))

(def ^:const ROW_MAJOR 101)
(def ^:const COLUMN_MAJOR 102)
(def ^:const DEFAULT_ORDER COLUMN_MAJOR)

(def ^:const NO_TRANS 111)
(def ^:const TRANS 112)
(def ^:const CONJ_TRANS 113)
(def ^:const DEFAULT_TRANS NO_TRANS)

(def ^:const UPPER 121)
(def ^:const LOWER 122)
(def ^:const DEFAULT_UPLO LOWER)

(def ^:const DIAG_NON_UNIT 131)
(def ^:const DIAG_UNIT 132)
(def ^:const DEFAULT_DIAG DIAG_NON_UNIT)

(def ^:const LEFT 141)
(def ^:const RIGHT 142)

(defn dec-property
  [^long code]
  (case code
    101 :row
    102 :column
    111 :no-trans
    112 :trans
    113 :conj-trans
    114 :atlas-conj
    121 :upper
    122 :lower
    131 :non-unit
    132 :unit
    141 :left
    142 :right
    :unknown))

(defn enc-property [option]
  (case option
    :row 101
    :column 102
    :no-trans 111
    :trans 112
    :conj-trans 113
    :atlas-conj 114
    :upper 121
    :lower 122
    :non-unit 131
    :unit 132
    :left 141
    :right 142
    nil))

(defn enc-order ^long [order]
  (if (= :row order) 101 102))

(defn enc-uplo ^long [uplo]
  (if (= :upper uplo) 121 122))

(defn enc-diag ^long [diag]
  (if (= :unit diag) 132 131))

(def ^{:no-doc true :const true} MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def ^{:no-doc true :const true} VECTOR_BOUNDS_MSG
  "Requested dim %d is out of bounds of vector dim %d.")

(def ^{:no-doc true :const true} INCOMPATIBLE_BLOCKS_MSG
  "Operation is not permited on objects with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  \n1: %s
  \n2: %s")

(def ^{:no-doc true :const true} INCOMPATIBLE_BLOCKS_MSG_3
  "Operation is not permited on objects with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  \n1: %s
  \n2: %s
  \n3: %s")

(def ^{:no-doc true :const true} INCOMPATIBLE_BLOCKS_MSG_4
  "Operation is not permited on objects with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  \n1: %s
  \n2: %s
  \n3: %s
  \n4: %s")

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

(def ^{:no-doc true :const true} ROTM_COND_MSG
  "Arguments x and y must be compatible and have equeal dimensions.
  argument param must have dimension at least 5.
  \nx: %s;
  \ny: %s;
  \nparam: %s")

(def ^{:no-doc true :const true} ROTMG_COND_MSG
  "Arguments d1d2xy and param must be compatible.
  param must have dimension at least 5 and d1d2xy at least 4.
  \nd1d2xy: %s;
  \nparam: %s ")
