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
           [clojure.lang Seqable IFn IFn$DD IFn$LD IFn$LDD IFn$LLD IFn$L]
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
(declare ->RealTriangularMatrix)
(declare real-ge-matrix)

(deftype RealGeneralMatrix [index* set* col* row*
                            ^uncomplicate.neanderthal.protocols.Factory fact
                            ^RealBufferAccessor accessor ^BLAS eng
                            ^Class entry-type ^Boolean master
                            ^ByteBuffer buf ^long m ^long n
                            ^long fd ^long sd ^long ld ^long ord ^long tra]
  Object
  (hashCode [this]
    (matrix-fold this hash* (-> (hash :RealGeneralMatrix) (hash-combine m) (hash-combine n)
                                (hash-combine ord) (hash-combine tra))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (compatible a b) (= m (.mrows ^Matrix b)) (= n (.ncols ^Matrix b)))
      (= 0.0 (matrix-foldmap a p- p+ 0.0 b))
      :default false))
  (toString [this]
    (format "#RealGeneralMatrix[%s, ord:%s, tra:%s, mxn:%dx%d, ld:%d]"
            entry-type (dec-property ord) (dec-property tra)
            m n ld))
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
    (real-ge-matrix fact m n ord tra))
  (raw [this fact]
    (real-ge-matrix fact m n ord tra))
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
  (subtriangle [_ uplo diag];;TODO use real-tr-matrix
    (let [k (min m n)
          b (.slice accessor buf 0 (+ (* ld (dec k)) k))]
      (->RealTriangularMatrix fact accessor (tr-matrix-engine fact) entry-type
                              false b k ld ord uplo diag tra)))
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
  (trans [_]
    tra)
  (count [_]
    (* m n))
  Seqable
  (seq [a];;TODO maybe seq should return a sequence of entries
    (map #(seq (.col a %)) (range 0 fd)))
  IFn$LLD
  (invokePrim [a i j] ;;TODO extract with-check-bounds and use it in core too
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
    (set* accessor buf ld fd val))
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
                    k l ld ord tra))
  (transpose [a]
    (real-ge-matrix fact false buf n m ld ord (if (= NO_TRANS tra) TRANS NO_TRANS))))

(defn real-block-vector [fact master buf n strd] ;;TODO move to the top
  (let [accessor (data-accessor fact)]
    (RealBlockVector. fact accessor (vector-engine fact) (.entryType accessor)
                      master buf n strd)))

(let [column-index (fn ^long [^long ld ^long i ^long j]
                     (+ (* ld j) i))
      row-index (fn ^long [^long ld ^long i ^long j]
                  (+ (* ld i) j))
      init-all (fn [^DataAccessor accessor buf _ _ val]
                 (.initialize accessor buf val))
      init-slice (fn [accessor buf ld fd val]
                   (dotimes [i fd]
                     (.initialize ^DataAccessor accessor
                                  (.slice ^BufferAccessor accessor
                                          buf (* (long ld) i) fd) val)))
      straight-cut (fn [fact ^ByteBuffer buf ld sd fd idx]
                     (real-block-vector fact false
                                        (.slice ^BufferAccessor (data-accessor fact)
                                                buf (* (long ld) (long idx)) sd)
                                        sd 1))

      cross-cut (fn [fact ^ByteBuffer buf ld sd fd idx]
                  (real-block-vector fact false
                                     (.slice ^BufferAccessor (data-accessor fact)
                                             buf idx (inc (* (dec (long fd)) (long ld)))) fd ld))]

  (defn real-ge-matrix
    ([fact master buf m n ld ord tra]
     (let [columnar (columnar? ord tra)
           sd (long (if columnar m n))
           fd (long (if columnar n m))
           ld (max sd (long ld))
           accessor (data-accessor fact)]
       (RealGeneralMatrix. (if columnar column-index row-index)
                           (if (= ld sd) init-all init-slice)
                           (if columnar straight-cut cross-cut)
                           (if columnar cross-cut straight-cut)
                           fact accessor (matrix-engine fact)
                           (.entryType accessor) master
                           buf m n fd sd ld ord tra)))
    ([fact m n ord tra]
     (let-release [buf (.createDataSource (data-accessor fact) (* (long m) (long n)))]
       (real-ge-matrix fact true buf m n 0 ord tra)))
    ([fact m n]
     (real-ge-matrix fact m n DEFAULT_ORDER DEFAULT_TRANS))))

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
        d (* m n)]
    (loop [i 0 src source]
      (if (and src (< i d))
        (do
          (if (column-major? destination)
            (.set destination (rem i m) (quot i m) (first src))
            (.set destination (rem i n) (quot i n) (first src)))
          (recur (inc i) (next src)))
        destination))))

(defmethod print-method RealGeneralMatrix
  [^RealGeneralMatrix a ^java.io.Writer w]
  (.write w (format "%s%s" (str a) (pr-str (seq a)))))

;; =================== Real Triangular Matrix ==================================

(deftype RealTriangularMatrix [^uncomplicate.neanderthal.protocols.Factory fact
                               ^RealBufferAccessor accessor ^BLAS eng
                               ^Class entry-type ^Boolean master
                               ^ByteBuffer buf ^long n ^long ld
                               ^long ord ^long fuplo ^long fdiag ^long tra]
  Object
  (hashCode [this]
    (matrix-fold this hash*;;TODO check fluokitten
                 (-> (hash :RealTriangularMatrix) (hash-combine n)
                     (hash-combine fuplo) (hash-combine fdiag) (hash-combine tra))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (compatible a b) (= n (.mrows ^Matrix b)) (= n (.ncols ^Matrix b)))
      (= 0.0 (matrix-foldmap a p- p+ 0.0 b));;TODO check fluokitten
      :default false))
  (toString [a]
    (format "#RealTriangularMatrix[%s, ord:%s, uplo:%s, diag:%s, tra:%s, mxn:%dx%d, ld:%d]"
            entry-type (dec-property ord) (dec-property fuplo)
            (dec-property fdiag) (dec-property tra)
            n n ld))
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
    (create-tr-matrix fact n (.createDataSource accessor (* n n))
                      {:order ord :uplo fuplo :diag fdiag :trans tra}))
  (raw [_ fact]
    (create-tr-matrix fact n (.createDataSource (data-accessor fact) (* n n))
                      {:order ord :uplo fuplo :diag fdiag :trans tra}))
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
    (create-tr-matrix fact 0 (.createDataSource accessor 0) nil))
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
  (order [_]
    ord)
  (uplo [_]
    fuplo)
  (diag [_]
    fdiag)
  (trans [_]
    tra)
  Seqable;;TODO the following few are the same as GE. Extract functions perhaps?
  (seq [a]
    (if (column-major? a)
      (map #(seq (.col a %)) (range 0 n))
      (map #(seq (.row a %)) (range 0 n))))
  IFn$LLD
  (invokePrim [a i j]
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
  RealChangeable;;TODO (see previous TODO) END of same functionality as GE
  (set [a val]
    (if (= ld n)
      (.initialize accessor buf val)
      (if (column-major? a)
        (dotimes [i n]
          (.set ^RealChangeable (.col a i) val))
        (dotimes [i n]
          (.set ^RealChangeable (.row a i) val))))
    a)
  (set [a i j val]
    (if (or (and (= COLUMN_MAJOR ord) (= UPPER fuplo))
            (and (= ROW_MAJOR ord) (= LOWER fuplo)))
      (.set accessor buf (+ (* ld (max i j)) (min i j)) val)
      (.set accessor buf (+ (* ld (min i j)) (max i j)) val))
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a i j f]
    (.set a i j (.invokePrim ^IFn$DD f (.entry a i j))))
  RealMatrix
  (mrows [_]
    n)
  (ncols [_]
    n)
  (entry [a i j]
    (if ((if (= UPPER fuplo) <= >=) i j)
      (if (= COLUMN_MAJOR ord)
        (.get accessor buf (+ (* ld j) i))
        (.get accessor buf (+ (* ld i) j)))
      0.0))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (throw (UnsupportedOperationException. "Cannot access parts of a TR matrix.")))
  (col [a j]
    (throw (UnsupportedOperationException. "Cannot access parts of a TR matrix.")))
  (submatrix [a i j k l]
    (throw (UnsupportedOperationException. "Cannot access parts of a TR matrix.")))
  (transpose [a]
    (RealTriangularMatrix. fact accessor eng entry-type false
                           buf n ld ord fuplo fdiag
                           (if (= NO_TRANS tra) TRANS NO_TRANS))))

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
          (if (column-major? destination)
            (.set destination (rem i m) (quot i m) (first src))
            (.set destination (rem i n) (quot i n) (first src)))
          (recur (inc i) (next src)))
        destination))))

(defmethod print-method RealTriangularMatrix
  [^RealTriangularMatrix a ^java.io.Writer w]
  (.write w (format "%s%s" (str a) (pr-str (seq a)))))
