;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.impl.buffer-block
  (:refer-clojure :exclude [accessor])
  (:require [vertigo
             [core :refer [wrap]]
             [bytes :refer [direct-buffer byte-seq slice-buffer]]
             [structs :refer [float64 float32 wrap-byte-seq]]]
            [uncomplicate.fluokitten.protocols
             :refer [PseudoFunctor Functor Foldable Magma Monoid Applicative]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [core :refer [transfer! copy!]]]
            [uncomplicate.neanderthal.impl.fluokitten :refer :all]
            [uncomplicate.commons.core :refer [Releaseable release let-release]])
  (:import [java.nio ByteBuffer DirectByteBuffer]
           [clojure.lang Seqable IFn IFn$DD IFn$LD IFn$LDD IFn$LLD IFn$L IFn$LLL]
           [vertigo.bytes ByteSeq]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus RealBufferAccessor BufferAccessor DataAccessor
            Vector Matrix RealVector RealMatrix RealChangeable Block]))

(defn ^:private hash* ^double [^double h ^double x]
  (double (clojure.lang.Util/hashCombine h (Double/hashCode x))))

(defn ^:private p- ^double [^double x ^double y]
  (- x y))

(defn ^:private p+ ^double [^double x ^double y]
  (+ x y))

;; ============ Realeaseable ===================================================

(defn ^:private clean-buffer [^ByteBuffer buffer]
  (do
    (if (.isDirect buffer)
      (.clean (.cleaner ^DirectByteBuffer buffer)))
    true))

;; ============ Real Buffer ====================================================

(deftype FloatBufferAccessor []
  DataAccessor
  (entryType [_]
    Float/TYPE)
  (entryWidth [_]
    Float/BYTES)
  (count [_ b]
    (quot (.capacity ^ByteBuffer b) Float/BYTES))
  (createDataSource [_ n]
    (direct-buffer (* Float/BYTES n)))
  (initialize [_ b]
    b)
  (initialize [this b v]
    (let [v (double v)
          strd Float/BYTES]
      (dotimes [i (.count this b)]
        (.putFloat ^ByteBuffer b (* strd i) v))
      b))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible [_ o]
    (instance? FloatBufferAccessor (data-accessor o)))
  BufferAccessor
  (toSeq [this buf stride]
    (if (< 0 (.count this buf))
      (wrap-byte-seq float32 (* Float/BYTES stride) 0 (byte-seq buf))
      (list)))
  (slice [_ buf k l]
    (slice-buffer buf (* Float/BYTES k) (* Float/BYTES l)))
  RealBufferAccessor
  (get [_ buf i]
    (.getFloat buf (* Float/BYTES i)))
  (set [_ buf i val]
    (.putFloat buf (* Float/BYTES i) val)))

(def float-accessor (->FloatBufferAccessor))

(deftype DoubleBufferAccessor []
  DataAccessor
  (entryType [_]
    Double/TYPE)
  (entryWidth [_]
    Double/BYTES)
  (count [_ b]
    (quot (.capacity ^ByteBuffer b) Double/BYTES))
  (createDataSource [_ n]
    (direct-buffer (* Double/BYTES n)))
  (initialize [_ b]
    b)
  (initialize [this b v]
    (let [v (double v)
          strd Double/BYTES]
      (dotimes [i (.count this b)]
        (.putDouble ^ByteBuffer b (* strd i) v))
      b))
  DataAccessorProvider
  (data-accessor [this]
    this)
  MemoryContext
  (compatible [_ o]
    (instance? DoubleBufferAccessor (data-accessor o)))
  BufferAccessor
  (toSeq [this buf stride]
    (if (< 0 (.count this buf))
      (wrap-byte-seq float64 (* Double/BYTES stride) 0 (byte-seq buf))
      (list)))
  (slice [_ buf k l]
    (slice-buffer buf (* Double/BYTES k) (* Double/BYTES l)))
  RealBufferAccessor
  (get [_ buf i]
    (.getDouble buf (* Double/BYTES i)))
  (set [_ buf i val]
    (.putDouble buf (* Double/BYTES i) val)))

(def double-accessor (->DoubleBufferAccessor))

;; ============ Real Vector ====================================================

(deftype RealBlockVector [^uncomplicate.neanderthal.protocols.Factory fact
                          ^RealBufferAccessor accessor ^BLAS eng
                          ^Class entry-type ^Boolean master
                          ^ByteBuffer buf ^long n ^long strd]
  Object
  (hashCode [this]
    (vector-fold this hash* (-> (hash :RealBlockVector) (hash-combine n))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (compatible x y) (= n (.dim ^Vector y)))
      (= 0.0 (vector-reduce-map p+ 0.0 p- x y))
      :default false))
  (toString [_]
    (format "#RealBlockVector[%s, n:%d, stride:%d]" entry-type n strd))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  Seqable
  (seq [_]
    (.toSeq accessor buf strd))
  Container
  (raw [_]
    (create-vector fact n (.createDataSource accessor n) nil))
  (raw [_ fact]
    (create-vector fact n (.createDataSource (data-accessor fact) n) nil))
  (zero [this]
    (raw this))
  (zero [this fact]
    (raw this fact))
  (host [this]
    (let-release [res (raw this)]
      (.copy eng this res)))
  (native [this]
    this)
  Monoid
  (id [x]
    (create-vector fact 0 (.createDataSource accessor 0) nil))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    fact)
  DataAccessorProvider
  (data-accessor [_]
    accessor)
  MemoryContext
  (compatible [_ y]
    (and (instance? RealBlockVector y) (= entry-type (.entryType ^Block y))))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    buf)
  (offset [_]
    0)
  (stride [_]
    strd)
  (count [_]
    n)
  IFn$LD
  (invokePrim [x i]
    (.entry x i))
  IFn
  (invoke [x i]
    (.entry x i))
  (invoke [x]
    n)
  IFn$L
  (invokePrim [x]
    n)
  RealChangeable
  (set [x val]
    (if (= 0 strd)
      (.initialize accessor buf val)
      (dotimes [i n]
        (.set accessor buf (* strd i) val)))
    x)
  (set [x i val]
    (.set accessor buf (* strd i) val)
    x)
  (setBoxed [x val]
    (.set x val))
  (setBoxed [x i val]
    (.set x i val))
  (alter [x i f]
    (.set x i (.invokePrim ^IFn$DD f (.entry x i))))
  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (.get accessor buf (* strd i)))
  (boxedEntry [x i]
    (.entry x i))
  (subvector [_ k l]
    (let [b (.slice accessor buf (* k strd) (inc (* (dec l) strd)))]
      (RealBlockVector. fact accessor eng entry-type false b l strd))))

(defn real-block-vector [fact master buf n strd]
  (let [accessor (data-accessor fact)]
    (RealBlockVector. fact accessor (vector-engine fact) (.entryType accessor)
                      master buf n strd)))

(extend RealBlockVector
  PseudoFunctor
  {:fmap! vector-fmap!}
  Functor
  {:fmap vector-fmap}
  Foldable
  {:fold vector-fold
   :foldmap vector-foldmap}
  Applicative
  {:pure vector-pure}
  Magma
  {:op vector-op})

(defmethod transfer! [RealBlockVector RealBlockVector]
  [source destination]
  (do
    (copy! source destination)
    destination))

(defmethod transfer! [clojure.lang.Sequential RealBlockVector]
  [source ^RealBlockVector destination]
  (let [n (.dim destination)]
    (do
      (loop [i 0 src source]
        (if (and src (< i n))
          (do
            (.set destination i (first src))
            (recur (inc i) (next src)))
          destination)))))

(defmethod print-method RealBlockVector
  [^Vector x ^java.io.Writer w]
  (.write w (format "%s%s" (str x) (pr-str (take 100 (seq x))))))

;; =================== Real Matrix =============================================

(declare real-ge-matrix)
(declare real-tr-matrix)

(defn ^:private straight-cut [fact ^ByteBuffer buf ld sd fd idx]
  (real-block-vector fact false
                     (.slice ^BufferAccessor (data-accessor fact)
                             buf (* (long ld) (long idx)) sd)
                     sd 1))

(defn ^:private cross-cut [fact ^ByteBuffer buf ld sd fd idx]
  (real-block-vector fact false
                     (.slice ^BufferAccessor (data-accessor fact)
                             buf idx (inc (* (dec (long fd)) (long ld))))
                     fd ld))

(deftype RealGeneralMatrix [index* col* row*
                            ^uncomplicate.neanderthal.protocols.Factory fact
                            ^RealBufferAccessor accessor ^BLAS eng
                            ^Class entry-type ^Boolean master
                            ^ByteBuffer buf ^long m ^long n
                            ^long ld ^long sd ^long fd ^long ord]
  Object
  (hashCode [this]
    (matrix-fold this hash* (-> (hash :RealGeneralMatrix) (hash-combine m)
                                (hash-combine n) (hash-combine ord))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (compatible a b) (= m (.mrows ^Matrix b)) (= n (.ncols ^Matrix b)))
      (= 0.0 (matrix-foldmap a p- p+ 0.0 b))
      :default false))
  (toString [this]
    (format "#RealGeneralMatrix[%s, mxn:%dx%d, ld:%d, ord%s]"
            entry-type m n ld (dec-property ord)))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    fact)
  DataAccessorProvider
  (data-accessor [_]
    accessor)
  Container
  (raw [_]
    (real-ge-matrix fact m n ord))
  (raw [this fact]
    (real-ge-matrix fact m n ord))
  (zero [this]
    (raw this))
  (zero [this fact]
    (raw this fact))
  (host [this]
    (let-release [res (raw this)]
      (.copy eng this res)))
  (native [this]
    this)
  DenseContainer
  (is-ge [_]
    true)
  (subtriangle [_ uplo diag];;TODO remove and introduce new function similar to copy that reuses memory
    (let [k (min m n)
          b (.slice accessor buf 0 (+ (* ld (dec k)) k))]
      (real-tr-matrix fact false b k ld ord uplo diag)))
  Monoid
  (id [a]
    (real-ge-matrix fact 0 0))
  MemoryContext
  (compatible [_ b]
    (and (or (instance? RealGeneralMatrix b) (instance? RealBlockVector b))
         (= entry-type (.entryType ^Block b))))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    buf)
  (offset [_]
    0)
  (stride [_]
    ld)
  (order [_]
    ord)
  (count [_]
    (* m n))
  Seqable
  (seq [a]
    (map #(seq (straight-cut fact buf ld sd fd %)) (range 0 fd)))
  IFn$LLD
  (invokePrim [a i j] ;;TODO extract with-check-bounds and use it in core too or call core/entry here!
    (if (and (< -1 i m) (< -1 j n))
      (.entry a i j)
      (throw (IndexOutOfBoundsException. (format MAT_BOUNDS_MSG i j m n)))))
  IFn
  (invoke [a i j]
    (if (and (< -1 (long i) m) (< -1 (long j) n))
      (.entry a i j)
      (throw (IndexOutOfBoundsException. (format MAT_BOUNDS_MSG i j m n)))))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (set [a val]
    (dotimes [j fd]
      (dotimes [i sd]
        (.set accessor buf (+ (* ld j) i) val)))
    a)
  (set [a i j val]
    (.set accessor buf (index* ld i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a i j f]
    (let [ind (index* ld i j)]
      (.set accessor buf ind (.invokePrim ^IFn$DD f (.get accessor buf ind)))
      a))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [a i j]
    (.get accessor buf (index* ld i j)))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (row* fact buf ld sd fd i))
  (col [a j]
    (col* fact buf ld sd fd j))
  (submatrix [a i j k l]
    (real-ge-matrix fact false
                    (.slice accessor buf (index* ld i j) (index* ld k (dec l)))
                    k l ld ord))
  (transpose [a]
    (real-ge-matrix fact false buf n m ld (if (= COLUMN_MAJOR ord) ROW_MAJOR COLUMN_MAJOR))))

(defn no-trans-index ^long [^long ld ^long i ^long j]
  (+ (* ld j) i))

(defn trans-index ^long [^long ld ^long i ^long j]
  (+ (* ld i) j))

(defn real-ge-matrix
  ([fact master buf m n ld ord]
   (let [no-trans (= COLUMN_MAJOR ord)
         sd (long (if no-trans m n))
         ld (max sd (long ld))
         fd (long (if no-trans n m))
         accessor (data-accessor fact)]
     (RealGeneralMatrix. (if no-trans no-trans-index trans-index)
                         (if no-trans straight-cut cross-cut)
                         (if no-trans cross-cut straight-cut)
                         fact accessor (matrix-engine fact)
                         (.entryType accessor) master
                         buf m n ld sd fd ord)))
  ([fact m n ord]
   (let-derelease [buf (.createDataSource (data-accessor fact) (* (long m) (long n)))]
     (real-ge-matrix fact true buf m n 0 ord)))
  ([fact m n]
   (real-ge-matrix fact m n DEFAULT_ORDER)))

(extend RealGeneralMatrix
  PseudoFunctor
  {:fmap! matrix-fmap!}
  Functor
  {:fmap matrix-fmap}
  Foldable
  {:fold matrix-fold
   :foldmap matrix-foldmap}
  Applicative
  {:pure matrix-pure}
  Magma
  {:op matrix-op})

(defmethod transfer! [RealGeneralMatrix RealGeneralMatrix]
  [source destination]
  (do
    (copy! source destination)
    destination))

(defmethod transfer! [clojure.lang.Sequential RealGeneralMatrix]
  [source ^RealGeneralMatrix destination]
  (let [m (.mrows destination)
        n (.ncols destination)
        sd (if (= COLUMN_MAJOR (.order destination)) m n)
        d (* m n)]
    (loop [i 0 src source]
      (when (and src (< i d))
        (.set destination (rem i sd) (quot i sd) (first src))
        (recur (inc i) (next src))))
    destination))

(defmethod print-method RealGeneralMatrix
  [^RealGeneralMatrix a ^java.io.Writer w]
  (.write w (format "%s%s" (str a) (pr-str (seq a)))))

;; =================== Real Triangular Matrix ==================================

(deftype RealTriangularMatrix [index*
                               col* row*
                               ^IFn$LLL col-start* ^IFn$LLL col-len*
                               ^IFn$LLL row-start* ^IFn$LLL row-len*
                               ^long diag-pad
                               ^uncomplicate.neanderthal.protocols.Factory fact
                               ^RealBufferAccessor accessor ^BLAS eng
                               ^Class entry-type ^Boolean master
                               ^ByteBuffer buf ^long n ^long ld
                               ^long ord ^long fuplo ^long fdiag]
  Object
  (hashCode [this]
    (matrix-fold this hash*;;TODO check fluokitten
                 (-> (hash :RealTriangularMatrix) (hash-combine n)
                     (hash-combine ord) (hash-combine fuplo) (hash-combine fdiag))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (compatible a b) (= n (.mrows ^Matrix b)) (= n (.ncols ^Matrix b)))
      (= 0.0 (matrix-foldmap a p- p+ 0.0 b));;TODO check fluokitten
      :default false))
  (toString [a]
    (format "#RealTriangularMatrix[%s, mxn:%dx%d, ld:%d, ord%s, uplo%s, diag%s]"
            entry-type n n ld (dec-property ord) (dec-property fuplo) (dec-property fdiag)))
  Releaseable
  (release [_]
    (if master (clean-buffer buf) true))
  EngineProvider
  (engine [_]
    eng)
  FactoryProvider
  (factory [_]
    fact)
  (native-factory [_]
    fact)
  DataAccessorProvider
  (data-accessor [_]
    accessor)
  Container
  (raw [_]
    (real-tr-matrix fact n ord fuplo fdiag))
  (raw [_ fact]
    (real-tr-matrix fact n ord fuplo fdiag))
  (zero [this]
    (raw this))
  (zero [this fact]
    (raw this fact))
  (host [this]
    (let-release [res (raw this)]
      (.copy eng this res)))
  (native [this]
    this)
  DenseContainer
  (is-ge [_]
    false)
  Monoid
  (id [a]
    (real-tr-matrix fact 0))
  MemoryContext
  (compatible [_ b]
    (and (or (instance? RealTriangularMatrix b) (instance? RealGeneralMatrix b)
             (instance? RealBlockVector b))
         (= entry-type (.entryType ^Block b))))
  Block
  (entryType [_]
    entry-type)
  (buffer [_]
    buf)
  (offset [_]
    0)
  (stride [_]
    ld)
  (count [_]
    (* n n))
  (uplo [_]
    fuplo)
  (diag [_]
    fdiag)
  (order [_]
    ord)
  Seqable
  (seq [a]
    (map #(seq (straight-cut fact buf ld n n %)) (range 0 n)));;TODO uplo
  IFn$LLD
  (invokePrim [a i j];;TODO extract with-check-bounds
    (if (and (< -1 i n) (< -1 j n))
      (.entry a i j)
      (throw (IndexOutOfBoundsException. (format MAT_BOUNDS_MSG i j n n)))))
  IFn
  (invoke [a i j]
    (if (and (< -1 (long i) n) (< -1 (long j) n))
      (.entry a i j)
      (throw (IndexOutOfBoundsException. (format MAT_BOUNDS_MSG i j n n)))))
  (invoke [a]
    n)
  IFn$L
  (invokePrim [a]
    n)
  RealChangeable
  (set [a val]
    (let [right-side (if (= COLUMN_MAJOR ord) (= UPPER fuplo) (= LOWER fuplo))]
      (if right-side
        (dotimes [j n]
          (dotimes [i (- j diag-pad)]
            (.set accessor buf (+ (* ld (+ j diag-pad)) i) val)))
        (dotimes [j n]
          (dotimes [i (- j diag-pad)]
            (.set accessor buf (+ (* ld j) j i diag-pad) val))))))
  (set [a i j val]
    (.set accessor buf (index* ld i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a i j f]
    (let [ind (index* ld i j)]
      (.set accessor buf ind (.invokePrim ^IFn$DD f (.get accessor buf ind)))
      a))
  RealMatrix
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (.get accessor buf (index* ld i j)))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (.subvector ^Vector (row* fact buf ld n n i)
                (+ (long (row-start* n i)) diag-pad)
                (- (long (row-len* n i)) diag-pad)))
  (col [a j]
    (.subvector ^Vector (col* fact buf ld n n j)
                (+ (long (col-start* n j)) diag-pad)
                (- (long (col-len* n j)) diag-pad)))
  (submatrix [a i j k l]
    (throw (UnsupportedOperationException. "TODO I should return banded matrix or GE.")))
  (transpose [a]
    (real-tr-matrix fact false buf n ld (if (= COLUMN_MAJOR ord) ROW_MAJOR COLUMN_MAJOR) fuplo fdiag)))

(let [zero (fn ^long [^long n ^long i] 0)
      id (fn ^long [^long n ^long i] i)
      inc-id (fn ^long [^long n ^long i] (inc i))
      inv (fn ^long [^long n ^long i] (- n i))]

  (defn real-tr-matrix
    ([fact master buf n ld ord uplo diag]
     (let [no-trans (= COLUMN_MAJOR ord)
           left (if no-trans (= LOWER uplo) (= UPPER uplo))
           accessor (data-accessor fact)]
       (RealTriangularMatrix. (if no-trans no-trans-index trans-index)
                              (if no-trans straight-cut cross-cut)
                              (if no-trans cross-cut straight-cut)
                              (if left id zero)
                              (if left inv inc-id)
                              (if left zero id)
                              (if left inc-id inv)
                              (if (= DIAG_UNIT diag) 1 0)
                              fact accessor (tr-matrix-engine fact)
                              (.entryType accessor) master
                              buf n ld ord uplo diag)))
    ([fact n ord uplo diag]
     (let-release [buf (.createDataSource (data-accessor fact) (* (long n) (long n)))]
       (real-tr-matrix fact true buf n ord uplo diag)))
    ([fact n]
     (real-tr-matrix fact n DEFAULT_TRANS DEFAULT_UPLO DEFAULT_DIAG))))

(extend RealTriangularMatrix
  PseudoFunctor
  {:fmap! matrix-fmap!}
  Functor
  {:fmap matrix-fmap}
  Foldable
  {:fold matrix-fold
   :foldmap matrix-foldmap}
  Applicative
  {:pure matrix-pure}
  Magma
  {:op matrix-op})

(defmethod transfer! [RealTriangularMatrix RealTriangularMatrix]
  [source destination]
  (do
    (copy! source destination)
    destination))

;;TODO
(defmethod transfer! [clojure.lang.Sequential RealTriangularMatrix]
  [source ^RealTriangularMatrix destination]
  (let [m (.mrows destination)
        n (.ncols destination)
        d (* m n)]
    (loop [i 0 src source]
      (if (and src (< i d))
        (do
          (if (= COLUMN_MAJOR (.order destination));;TODO check
            (.set destination (rem i m) (quot i m) (first src))
            (.set destination (rem i n) (quot i n) (first src)))
          (recur (inc i) (next src)))
        destination))))

(defmethod print-method RealTriangularMatrix
  [^RealTriangularMatrix a ^java.io.Writer w]
  (.write w (format "%s%s" (str a) (pr-str (seq a)))))
