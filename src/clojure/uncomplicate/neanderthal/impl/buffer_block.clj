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
             [core :refer [transfer! copy! col row mrows ncols]]
             [real :refer [sum]]]
            [uncomplicate.neanderthal.impl.fluokitten :refer :all]
            [uncomplicate.commons.core :refer [Releaseable release let-release]])
  (:import [java.nio ByteBuffer DirectByteBuffer]
           [clojure.lang Seqable IFn IFn$DD IFn$DDD IFn$DDDD IFn$DDDDD IFn$OLLDO
            IFn$LD IFn$LDD IFn$LLD IFn$L IFn$LLL IFn$OLLD IFn$OLLDD IFn$LLLL]
           [vertigo.bytes ByteSeq]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus RealBufferAccessor BufferAccessor DataAccessor
            Vector Matrix RealVector RealMatrix RealChangeable Block]))

(defn ^:private hash* ^double [^double h ^double x]
  (double (clojure.lang.Util/hashCombine h (Double/hashCode x))))

(defn ^:private consistent? [^Block a ^Block b]
  (and (instance? (class a) b) (compatible (data-accessor a) b )))

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
  (compatible [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? FloatBufferAccessor da))))
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
  (compatible [this o]
    (let [da (data-accessor o)]
      (or (identical? this da) (instance? DoubleBufferAccessor da))))
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
                          ^RealBufferAccessor accessor ^BLASPlus eng
                          ^Class entry-type ^Boolean master
                          ^ByteBuffer buf ^long n ^long strd]
  Object
  (hashCode [x]
    (vector-fold* hash* (-> (hash :RealBlockVector) (hash-combine n))
                  (.subvector x 0 (min 16 n))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? RealBlockVector y) (compatible accessor y) (= n (.dim ^Vector y)))
      (loop [i 0]
        (if (< i n)
          (if (= (.entry x i) (.entry ^RealVector y i))
            (recur (inc i))
            false)
          true))
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
  (form [this]
    :vector)
  MemoryContext
  (compatible [_ y]
    (compatible accessor y))
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
      (RealBlockVector. fact accessor eng entry-type false b l strd)))
  PseudoFunctor
  (fmap! [x f]
    (vector-fmap* ^IFn$DD f x))
  (fmap! [x f y]
    (vector-fmap* ^IFn$DDD f x y))
  (fmap! [x f y z]
    (vector-fmap* ^IFn$DDDD f x y z))
  (fmap! [x f y z v]
    (vector-fmap* ^IFn$DDDDD f x y z v))
  (fmap! [x f y z v ws]
    (vector-fmap* f x y z v ws))
  Foldable
  (fold [x]
    (.sum eng x))
  (fold [x f init]
    (vector-fold* f init x))
  (fold [x f init y]
    (vector-fold* f init x y))
  (fold [x f init y z]
    (vector-fold* f init x y z))
  (fold [x f init y z v]
    (vector-fold* f init x y z v))
  (fold [x f init y z v ws]
    (vector-fold* f init x y z v ws))
  (foldmap [x g]
    (loop [i 0 acc 0.0]
      (if (< i n)
        (recur (inc i) (+ acc (.invokePrim ^IFn$DD g (.entry x i))))
        acc)))
  (foldmap [x g f init]
    (vector-foldmap* f init ^IFn$DD g x))
  (foldmap [x g f init y]
    (vector-foldmap* f init ^IFn$DDD g x y))
  (foldmap [x g f init y z]
    (vector-foldmap* f int ^IFn$DDDD g x y z))
  (foldmap [x g f init y z v]
    (vector-foldmap* f init ^IFn$DDDDD g x y z v))
  (foldmap [x g f init y z v ws]
    (vector-foldmap* f init g x y z v ws)))()

(defn real-block-vector [fact master buf n strd]
  (let [accessor (data-accessor fact)]
    (RealBlockVector. fact accessor (vector-engine fact) (.entryType accessor)
                      master buf n strd)))

(extend RealBlockVector
  Functor
  {:fmap copy-fmap}
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

(defn ^:private no-trans-get ^double [^RealMatrix a ^long i ^long j]
  (.entry a i j))

(defn ^:private trans-get ^double [^RealMatrix a ^long i ^long j]
  (.entry a j i))

(defn ^:private no-trans-set [^RealChangeable a ^long i ^long j ^double val]
  (.set a i j val))

(defn ^:private trans-set [^RealChangeable a ^long i ^long j ^double val]
  (.set a j i val))

(deftype RealGeneralMatrix [^IFn$LLLL index* ^IFn$OLLD get* ^IFn$OLLDO set*
                            col-row* col* row*
                            ^uncomplicate.neanderthal.protocols.Factory fact
                            ^RealBufferAccessor accessor ^BLAS eng
                            ^Class entry-type ^Boolean master
                            ^ByteBuffer buf ^long m ^long n
                            ^long ld ^long sd ^long fd ^long ord]
  Object
  (hashCode [this]
    (matrix-fold* fd col-row* hash*
                  (-> (hash :RealGeneralMatrix) (hash-combine m) (hash-combine n))
                  (.submatrix this 0 0 (min 16 m) (min 16 n))))
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? RealGeneralMatrix b) (compatible accessor b)
           (= m (mrows b)) (= n (ncols b)))
      (loop [j 0]
        (if (< j fd)
          (and (loop [i 0]
                 (if (< i sd)
                   (and (= (.get accessor buf (+ (* ld j) i))
                           (.invokePrim get* ^RealGeneralMatrix b i j))
                        (recur (inc i)))
                   true))
               (recur (inc j)))
          true))
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
  (form [this]
    :GE)
  MemoryContext
  (compatible [_ b]
    (compatible accessor b))
  DenseContainer
  (subtriangle [_ uplo diag];;TODO remove and introduce new function similar to copy that reuses memory
    (let [k (min m n)
          b (.slice accessor buf 0 (+ (* ld (dec k)) k))]
      (real-tr-matrix fact false b k ld ord uplo diag)))
  Monoid
  (id [a]
    (real-ge-matrix fact 0 0))
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
    (.set accessor buf (.invokePrim index* ld i j) val)
    a)
  (setBoxed [a val]
    (.set a val))
  (setBoxed [a i j val]
    (.set a i j val))
  (alter [a i j f]
    (.set a i j (.invokePrim ^IFn$DD f (.entry a i j))))
  RealMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [a i j]
    (.get accessor buf (.invokePrim index* ld i j)))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (row* fact buf ld sd fd i))
  (col [a j]
    (col* fact buf ld sd fd j))
  (submatrix [a i j k l]
    (real-ge-matrix fact false
                    (.slice accessor buf (.invokePrim index* ld i j) (.invokePrim index* ld k (dec l)))
                    k l ld ord))
  (transpose [a]
    (real-ge-matrix fact false buf n m ld (if (= COLUMN_MAJOR ord) ROW_MAJOR COLUMN_MAJOR)))
  PseudoFunctor
  (fmap! [a f]
    (matrix-fmap* set* get* fd sd ^IFn$DD f a))
  (fmap! [a f b]
    (matrix-fmap* set* get* fd sd ^IFn$DDD f a b))
  (fmap! [a f b c]
    (matrix-fmap* set* get* fd sd ^IFn$DDDD f a b c))
  (fmap! [a f b c d]
    (matrix-fmap* set* get* fd sd ^IFn$DDDDD f a b c d))
  (fmap! [a f b c d es]
    (matrix-fmap* set* get* fd sd f a b c d nil))
  Foldable
  (fold [_]
    (loop [j 0 acc 0.0]
      (if (< j fd)
        (recur (inc j)
               (double
                (loop [i 0 acc acc]
                  (if (< i sd)
                    (recur (inc i) (+ acc (.get accessor buf (+ (* ld j) i))))
                    acc))))
        acc)))
  (fold [a f init]
    (matrix-fold* fd col-row* f init a))
  (fold [a f init b]
    (matrix-fold* fd col-row* f init a b))
  (fold [a f init b c]
    (matrix-fold* fd col-row* f init a b c))
  (fold [a f init b c d]
    (matrix-fold* fd col-row* f init a b c d))
  (fold [a f init b c d es]
    (matrix-fold* fd col-row* f init a b c d es))
  (foldmap [_ g]
    (loop [j 0 acc 0.0]
      (if (< j fd)
        (recur (inc j)
               (double
                (loop [i 0 acc acc]
                  (if (< i sd)
                    (recur (inc i)
                           (+ acc (.invokePrim ^IFn$DD g (.get accessor buf (+ (* ld j) i)))))
                    acc))))
        acc)))
  (foldmap [a g f init]
    (matrix-foldmap* fd col-row* f init ^IFn$DD g a))
  (foldmap [a g f init b]
    (matrix-foldmap* fd col-row* f init ^IFn$DDD g a b))
  (foldmap [a g f init b c]
    (matrix-foldmap* fd col-row* f init ^IFn$DDDD g a b c))
  (foldmap [a g f init b c d]
    (matrix-foldmap* fd col-row* f init ^IFn$DDDDD g a b c d))
  (foldmap [a g f init b c d es]
    (matrix-foldmap* fd col-row* f init g a b c d es)))

(defn no-trans-index ^long [^long ld ^long i ^long j]
  (+ (* ld j) i))

(defn trans-index ^long [^long ld ^long i ^long j]
  (+ (* ld i) j))

(defn real-ge-matrix
  ([fact master buf m n ld ord]
   (let [accessor (data-accessor fact)]
     (if (= COLUMN_MAJOR ord)
       (RealGeneralMatrix. no-trans-index no-trans-get no-trans-set col
                           straight-cut cross-cut fact accessor (matrix-engine fact)
                           (.entryType accessor) master buf
                           m n (max (long ld) (long m)) m n ord)
       (RealGeneralMatrix. trans-index trans-get trans-set row cross-cut
                           straight-cut fact accessor (matrix-engine fact)
                           (.entryType accessor) master buf
                           m n (max (long ld) (long n)) n m ord))))
  ([fact m n ord]
   (let-release [buf (.createDataSource (data-accessor fact) (* (long m) (long n)))]
     (real-ge-matrix fact true buf m n 0 ord)))
  ([fact m n]
   (real-ge-matrix fact m n DEFAULT_ORDER)))

(extend RealGeneralMatrix
  Functor
  {:fmap copy-fmap}
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

(defn left-index ^long [^long diag-pad ^long ld ^long i ^long j]
  (+ (* (inc ld) j) (+ i diag-pad)))

(defn right-index ^long [^long diag-pad ^long ld ^long i ^long j]
  (+ (* ld (+ j diag-pad)) i))

(deftype RealTRMatrix [^IFn$LLLL index* ^IFn$OLLD get* ^IFn$OLLDD set*
                       ^IFn$LLL start* ^IFn$LLL end*
                       col-row* col* row*
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
    #_(tr-matrix-fold* fd col-row* hash* (-> (hash :RealTRMatrix) (hash-combine n))
                     (.submatrix this 0 0 (min 16 n) (min 16 n))));;TODO check fluokitten and triangle and do with submatrix
  (equals [a b]
    (cond
      (nil? b) false
      (identical? a b) true
      (and (instance? RealTRMatrix b) (compatible accessor b)
           (= n (mrows b))
           (if (= fuplo (.uplo ^Block b))
             (= fdiag (.diag ^Block b))
             (not (= fdiag (.diag ^Block b)))))
      (loop [j 0]
        (if (< j n)
          (let [end (.invokePrim end* n j)]
            (and (loop [i (.invokePrim start* n j)]
                   (if (< i end)
                     (and (= (.get accessor buf (+ (* ld j) i))
                             (.invokePrim get* ^RealTRMatrix b i j))
                          (recur (inc i)))
                     true))
                 (recur (inc j))))
          true))
      :default false))
  (toString [a]
    (format "#RealTRMatrix[%s, mxn:%dx%d, ld:%d, ord%s, uplo%s, diag%s]"
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
      (.copy eng this res)));;implement TR.copy
  (native [this]
    this)
  (form [this]
    :TR)
  MemoryContext
  (compatible [_ b]
    (compatible accessor b))
  Monoid
  (id [a]
    (real-tr-matrix fact 0))
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
    (dotimes [j n]
      (let [end (.invokePrim end* n j)]
        (loop [i (.invokePrim start* n j)]
          (when (< i end)
            (.set accessor buf (+ (* ld j) i) val)))))
    a)
  (set [a i j val]
    (.set accessor buf (.invokePrim index* ld i j) val)
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
    (.get accessor buf (.invokePrim index* ld i j)))
  (boxedEntry [this i j]
    (.entry this i j))
  (row [a i]
    (.subvector ^Vector (row* fact buf ld n n i)
                (+ (.invokePrim row-start* n i) diag-pad)
                (- (.invokePrim row-len* n i) diag-pad)))
  (col [a j]
    (.subvector ^Vector (col* fact buf ld n n j)
                (+ (long (col-start* n j)) diag-pad)
                (- (long (col-len* n j)) diag-pad)))
  (submatrix [a i j k l]
    (throw (UnsupportedOperationException. "TODO I should return banded matrix or GE.")))
  (transpose [a]
    (real-tr-matrix fact false buf n ld (if (= COLUMN_MAJOR ord) ROW_MAJOR COLUMN_MAJOR) fuplo fdiag)))

(let [zero (fn ^long [^long n ^long i] 0)
      end (fn ^long [^long n ^long i] n)
      id (fn ^long [^long n ^long i] i)
      inc-id (fn ^long [^long n ^long i] (inc i))
      inv (fn ^long [^long n ^long i] (- n i))]

  (defn real-tr-matrix
    ([fact master buf n ld ord uplo diag]
     (let [no-trans (= COLUMN_MAJOR ord)
           non-unit (= DIAG_NON_UNIT diag)
           left (if no-trans (= LOWER uplo) (= UPPER uplo))
           accessor (data-accessor fact)]
       (RealTRMatrix. (if no-trans no-trans-index trans-index)
                      (if no-trans straight-cut cross-cut)
                      (if no-trans cross-cut straight-cut)
                      (if left (if non-unit id inc-id) zero)
                      (if left end (if non-unit inc-id id))
                      (if no-trans col row)
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


(defmethod transfer! [RealTRMatrix RealTRMatrix]
  [source destination]
  (do
    (copy! source destination)
    destination))

;;TODO
(defmethod transfer! [clojure.lang.Sequential RealTRMatrix]
  [source ^RealTRMatrix destination]
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

(defmethod print-method RealTRMatrix
  [^RealTRMatrix a ^java.io.Writer w]
  (.write w (format "%s%s" (str a) (pr-str (seq a)))))
