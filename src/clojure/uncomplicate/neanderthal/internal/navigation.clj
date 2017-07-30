;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.navigation
  (:require [uncomplicate.neanderthal.internal.api
             :refer [COLUMN_MAJOR ROW_MAJOR UPPER LOWER DIAG_NON_UNIT navigator region storage]])
  (:import [clojure.lang IFn$LLDD]
           [uncomplicate.neanderthal.internal.api RealOrderNavigator UploNavigator StripeNavigator
            LayoutNavigator RealLayoutNavigator DenseStorage FullStorage BandStorage
            Region RealDefault RealBufferAccessor
            BandNavigator RealChangeable RealMatrix Matrix]))

(deftype ColumnRealOrderNavigator []
  RealOrderNavigator
  (sd [_ m n]
    m)
  (fd [_ m n]
    n)
  (index [_ offset ld i j]
    (+ offset (* ld j) i))
  (index [_ offset ld k]
    (if (< 0 k)
      (+ offset (* ld k))
      (- offset k)))
  (get [_ a i j]
    (.entry ^RealMatrix a i j))
  (set [_ a i j val]
    (.set ^RealChangeable a i j val))
  (invokePrimitive [_ f i j val]
    (.invokePrim ^IFn$LLDD f i j val))
  (stripe [_ a j]
    (.col ^RealMatrix a j)))

(deftype RowRealOrderNavigator []
  RealOrderNavigator
  (sd [_ m n]
    n)
  (fd [_ m n]
    m)
  (index [_ offset ld i j]
    (+ offset (* ld i) j))
  (index [_ offset ld k]
    (if (< 0 k)
      (+ offset k)
      (- offset (* ld k))))
  (get [_ a i j]
    (.entry ^RealMatrix a j i))
  (set [_ a i j val]
    (.set ^RealChangeable a j i val))
  (invokePrimitive [_ f i j val]
    (.invokePrim ^IFn$LLDD f j i val))
  (stripe [_ a i]
    (.row ^RealMatrix a i)))

(def col-navigator (ColumnRealOrderNavigator.))
(def row-navigator (RowRealOrderNavigator.))

(deftype NonUnitLowerNavigator []
  UploNavigator
  (colStart [_ _ i]
    i)
  (colEnd [_ n _]
    n)
  (rowStart [_ _ _]
    0)
  (rowEnd [_ n i]
    (inc i))
  (diaDim [_ n k]
    (if (< k 1)
      (+ n k)
      0))
  (unitIndex [_ i]
    -1)
  (defaultEntry [_ i j]
    (* 2 (Long/signum (inc (Long/signum (- i j)))))))

(deftype UnitLowerNavigator []
  UploNavigator
  (colStart [_ _ i]
    (inc i))
  (colEnd [_ n _]
    n)
  (rowStart [_ _ _]
    0)
  (rowEnd [_ _ i]
    i)
  (diaDim [_ n k]
    (if (< k 0)
      (+ n k)
      0))
  (unitIndex [_ i]
    i)
  (defaultEntry [_ i j]
    (inc (Long/signum (- i j)))))

(deftype NonUnitUpperNavigator []
  UploNavigator
  (colStart [_ _ _]
    0)
  (colEnd [_ _ i]
    (inc i))
  (rowStart [_ _ i]
    i)
  (rowEnd [_ n _]
    n)
  (diaDim [_ n k]
    (if (< -1 k)
      (- n k)
      0))
  (unitIndex [_ i]
    -1)
  (defaultEntry [_ i j]
    (* 2 (Long/signum (inc (Long/signum (- j i)))))))

(deftype UnitUpperNavigator []
  UploNavigator
  (colStart [_ _ _]
    0)
  (colEnd [_ _ i]
    i)
  (rowStart [_ _ i]
    (inc i))
  (rowEnd [_ n _]
    n)
  (diaDim [_ n k]
    (if (< 0 k)
      (- n k)
      0))
  (unitIndex [_ i]
    i)
  (defaultEntry [_ i j]
    (inc (Long/signum (- j i)))))

(def non-unit-upper-nav (NonUnitUpperNavigator.))
(def unit-upper-nav (UnitUpperNavigator.))
(def non-unit-lower-nav (NonUnitLowerNavigator.))
(def unit-lower-nav (UnitLowerNavigator.))

(deftype UnitTopNavigator []
  StripeNavigator
  (start [_ _ _]
    0)
  (end [_ _ i]
    i)
  (offsetPad [_ ld]
    ld))

(deftype NonUnitTopNavigator []
  StripeNavigator
  (start [_ _ i]
    0)
  (end [_ _ i]
    (inc i))
  (offsetPad [_ _]
    0))

(deftype UnitBottomNavigator []
  StripeNavigator
  (start [_ _ i]
    (inc i))
  (end [_ n _]
    n)
  (offsetPad [_ _]
    1))

(deftype NonUnitBottomNavigator []
  StripeNavigator
  (start [_ _ i]
    i)
  (end [_ n _]
    n)
  (offsetPad [_ _]
    0))

(def non-unit-top-navigator (NonUnitTopNavigator.))
(def unit-top-navigator (UnitTopNavigator.))
(def non-unit-bottom-navigator (NonUnitBottomNavigator.))
(def unit-bottom-navigator (UnitBottomNavigator.))

(deftype ColumnBandNavigator []
  BandNavigator
  (height [_ _ _ kl ku]
    (inc (+ kl ku)))
  (width [_ m n _ ku]
    (min n (+ (min m n) ku)))
  (sd [_ m n kl _]
    (min m (+ (min m n) kl)))
  (fd [_ m n _ ku]
    (min n (+ (min m n) ku)))
  (start [_ _ ku j]
    (max 0 (- j ku)))
  (end [_ m _ kl _ j]
    (min m (inc (+ j kl))))
  (kd [_ _ ku]
    ku)
  (index [_ offset ld kl ku i j]
    (+ offset (* j ld) (- ku j) i))
  (index [this offset ld kl ku k]
    (.index this offset ld kl ku (max 0 (- k)) (max 0 k))))

(deftype RowBandNavigator []
  BandNavigator
  (height [_ m n kl _];;all these should be simple no-arg queries / maybe storage should be a replacement for info
    (min m (+ (min m n) kl)))
  (width [_ _ _ kl ku]
    (inc (+ kl ku)))
  (sd [_ m n _ ku]
    (min n (+ (min m n) ku)))
  (fd [_ m n kl _]
    (min m (+ (min m n) kl)))
  (start [_ kl _ i]
    (max 0 (- i kl)))
  (end [_ _ n _ ku i]
    (min n (inc (+ i ku))))
  (kd [_ kl _]
    kl)
  (index [_ offset ld kl _ i j]
    (+ offset (* i ld) (- kl i) j))
  (index [this offset ld kl ku k] ;; TODO this should delegate to the index in an abstract way for all storages
    (.index this offset ld kl ku (max 0 (- k)) (max 0 k))))

(def col-band-navigator (ColumnBandNavigator.))
(def row-band-navigator (RowBandNavigator.))

;; TODO new navigatio starts from here! ============================================


;; ==================== Band Region Selector =======================================

(deftype BandRegion [^long m ^long n ^long kl ^long ku]
  Object
  (hashCode [a]
    (-> (hash-combine m) (hash-combine n) (hash-combine kl) (hash-combine ku)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? BandRegion b)
             (= m (.m ^BandRegion b)) (= n (.n ^BandRegion b))
             (= (.kl ^BandRegion b)) (= (.ku ^BandRegion b)))))
  (toString [a]
    (format "#BandRegion[mxn:%dx%d, kl:%d, ku:%d]" m n kl ku))
  Region
  (accessible [_ i j]
    (<= (- j ku) i (+ j kl)))
  (colStart [_ j]
    (max 0 (- j ku)))
  (colEnd [_ j]
    (min m (inc (+ j kl))))
  (rowStart [_ i]
    (max 0 (- i kl)))
  (rowEnd [_ i]
    (min n (inc (+ i ku))))
  (isLower [_]
    (= m (inc kl)))
  (isUpper [_]
    (= n (inc ku)))
  (isDiagUnit [_]
    (or (= -1 kl) (= -1 ku))))

(defn band-region
  ([^long m ^long n ^long kl ^long ku]
   (BandRegion. m n kl ku))
  ([^long n lower? diag-unit?]
   (let [diag-pad (if diag-unit? -1 0)]
     (if lower?
       (BandRegion. n n (dec n) diag-pad)
       (BandRegion. n n diag-pad (dec n)))))
  ([^long n lower?]
   (if lower?
     (BandRegion. n n (dec n) 0)
     (BandRegion. n n 0 (dec n)))))

(defn transpose [^BandRegion region]
  (BandRegion. (.n region) (.m region) (.ku region) (.kl region)))

;; =================== Navigator ==========================================

(deftype RealColumnNavigator []
  LayoutNavigator
  (start [_ region j]
    (.colStart ^Region region j))
  (end [_ region j]
    (.colEnd ^Region region j))
  (stripe [_ a j]
    (.col ^Matrix a j))
  (isColumnMajor [_]
    true)
  (isRowMajor [_]
    false)
  RealLayoutNavigator
  (get [_ a i j]
    (.entry ^RealMatrix a i j))
  (set [_ a i j val]
    (.set ^RealChangeable a i j val))
  (invokePrimitive [_ f i j val]
    (.invokePrim ^IFn$LLDD f i j val)))

(deftype RealRowNavigator []
  LayoutNavigator
  (start [_ region j]
    (.rowStart ^Region region j))
  (end [_ region j]
    (.rowEnd ^Region region j))
  (stripe [_ a j]
    (.row ^Matrix a j))
  (isColumnMajor [_]
    false)
  (isRowMajor [_]
    true)
  RealLayoutNavigator
  (get [_ a i j]
    (.entry ^RealMatrix a j i))
  (set [_ a i j val]
    (.set ^RealChangeable a j i val))
  (invokePrimitive [_ f i j val]
    (.invokePrim ^IFn$LLDD f j i val)))

(def ^:const real-column-navigator (RealColumnNavigator.))
(def ^:const real-row-navigator (RealRowNavigator.))

;; =================== Full Storage ========================================

(deftype ColumnFullStorage [^long m ^long n ^long ld]
  FullStorage
  (sd [_]
    m)
  (ld [_]
    ld)
  DenseStorage
  (index [_ i j];;TODO offset in buffer block
    (+ (* j ld) i))
  (fd [_]
    n)
  (isColumnMajor [_]
    true)
  (layout [_]
    102))

(deftype RowFullStorage [^long m ^long n ^long ld]
  FullStorage
  (sd [_]
    n)
  (ld [_]
    ld)
  DenseStorage
  (index [_ i j];;TODO offset in buffer block
    (+ (* i ld) j))
  (fd [_]
    m)
  (isColumnMajor [_]
    false)
  (layout [_]
    101))

(defn full-storage
  ([^long layout ^long m ^long n ^long ld]
   (if (= COLUMN_MAJOR layout)
     (ColumnFullStorage. m n ld)
     (RowFullStorage. m n ld)))
  ([^long layout ^long m ^long n]
   (if (= COLUMN_MAJOR layout)
     (ColumnFullStorage. m n m)
     (RowFullStorage. m n n))))

;; =================== Full Band Storage =========================================

(deftype ColumnBandStorage [^long m ^long n ^long ld ^long kl ^long ku]
  BandStorage
  (height [_]
    (inc (+ kl ku)))
  (width [_]
    (min n (+ (min m n) ku)))
  (kd [_]
    ku)
  FullStorage
  (sd [_]
    (min m (+ (min m n) kl)))
  (ld [_]
    ld)
  DenseStorage
  (index [_ i j];;TODO offset in buffer block
    (+ (* j ld) (- ku j) i))
  (fd [_]
    (min n (+ (min m n) ku)))
  (isColumnMajor [_]
    true)
  (layout [_]
    102))

(deftype RowBandStorage [^long m ^long n ^long ld ^long kl ^long ku]
  BandStorage
  (height [_]
    (min m (+ (min m n) kl)))
  (width [_]
    (inc (+ kl ku)))
  (kd [_]
    kl)
  FullStorage
  (sd [_]
    (min n (+ (min m n) ku)))
  (ld [_]
    ld)
  DenseStorage
  (index [_ i j]
    (+ (* i ld) (- kl i) j))
  (fd [_]
    (min m (+ (min m n) kl)))
  (isColumnMajor [_]
    false)
  (layout [_]
    101))

(defn band-storage
  ([layout m n ld kl ku]
   (if (= COLUMN_MAJOR layout)
     (ColumnBandStorage. m n ld kl ku)
     (RowBandStorage. m n ld kl ku)))
  ([layout ^long n ^long uplo ^long diag]
   (let [unit-pad (if (= DIAG_NON_UNIT diag) 0 -1)]
     (if (= LOWER uplo)
       (band-storage layout n n (dec n) unit-pad)
       (band-storage layout n n unit-pad (dec n)))))
  ([layout ^long n ^long uplo]
   (if (= LOWER uplo)
     (band-storage layout n n (dec n) 0)
     (band-storage layout n n 0 (dec n)))))

;; =================== Packed Storage ======================================

(deftype UpperColumnPackedStorage [^long n]
  DenseStorage
  (index [_ i j]
    (+ i (unchecked-divide-int (* j (inc j)) 2)))
  (fd [_]
    n)
  (isColumnMajor [_]
    true)
  (layout [_]
    102))

(deftype UpperRowPackedStorage [^long n]
  DenseStorage
  (index [_ i j]
    (+ j (unchecked-divide-int (* i (dec (- (* 2 n) i))) 2)))
  (fd [_]
    n)
  (isColumnMajor [_]
    false)
  (layout [_]
    101))

(deftype LowerColumnPackedStorage [^long n]
  DenseStorage
  (index [_ i j]
    (+ i (unchecked-divide-int (* j (dec (- (* 2 n) j))) 2)))
  (fd [_]
    n)
  (isColumnMajor [_]
    true)
  (layout [_]
    102))

(deftype LowerRowPackedStorage [^long n]
  DenseStorage
  (index [_ i j]
    (+ j (unchecked-divide-int (* i (inc i)) 2)))
  (fd [_]
    n)
  (isColumnMajor [_]
    false)
  (layout [_]
    101))

(defn packed-storage [column? lower? ^long n]
  (if column?
    (if lower?
      (LowerColumnPackedStorage. n)
      (UpperColumnPackedStorage. n))
    (if lower?
      (LowerRowPackedStorage. n)
      (UpperRowPackedStorage. n))))

;; ======================= Layout utillities ==============================

(defmacro doall-layout
  ([nav stor region i j idx expr]
   `(dotimes [~j (.fd ~stor)]
      (let [start# (.start ~nav ~region ~j)
            end# (.end ~nav ~region ~j)]
        (dotimes [i# (- end# start#)]
          (let [~i (+ start# i#)
                ~idx (.index ~stor ~i ~j)]
            ~expr)))))
  ([nav stor region i j expr]
   `(dotimes [~j (.fd ~stor)]
      (let [start# (.start ~nav ~region ~j)
            end# (.end ~nav ~region ~j)]
        (dotimes [i# (- end# start#)]
          (let [~i (+ start# i#)]
            ~expr)))))
  ([a i j idx expr]
   `(let [nav# (navigator ~a)
          stor# (storage ~a)
          region# (region ~a)]
      (doall-layout nav# stor# region# ~i ~j ~idx ~expr)
      ~a))
  ([a i j expr]
   `(let [nav# (navigator ~a)
          stor# (storage ~a)
          region# (region ~a)]
      (doall-layout nav# stor# region# ~i ~j ~expr)
      ~a)))

(defmacro doseq-layout
  ([nav stor region i j idx source e expr]
   `(let [fd# (.fd ~stor)]
      (loop [~j 0 src# (seq ~source)]
        (if (and src# (< ~j fd#))
          (recur (inc ~j)
                 (let [end# (.end ~nav ~region ~j)]
                   (loop [~i (.start ~nav ~region ~j) src# src#]
                     (if (and src# (< ~i end#))
                       (do
                         (let [~idx (.index ~stor ~i ~j)
                               ~e (first src#)]
                           ~expr)
                         (recur (inc ~i) (next src#)))
                       src#))))
          ~source))))
  ([nav stor region i j source e expr]
   `(let [fd# (.fd ~stor)]
      (loop [~j 0 src# (seq ~source)]
        (if (and src# (< ~j fd#))
          (recur (inc ~j)
                 (let [end# (.end ~nav ~region ~j)]
                   (loop [~i (.start ~nav ~region ~j) src# src#]
                     (if (and src# (< ~i end#))
                       (do
                         (let [~e (first src#)]
                           ~expr)
                         (recur (inc ~i) (next src#)))
                       src#))))
          ~source))))
  ([a i j idx source e expr]
   `(let [nav# (navigator ~a)
          stor# (storage ~a)
          region# (region ~a)]
      (doseq-layout nav# stor# region# ~i ~j ~idx ~source ~e ~expr)
      ~a))
  ([a i j source e expr]
   `(let [nav# (navigator ~a)
          stor# (storage ~a)
          region# (region ~a)]
      (doseq-layout nav# stor# region# ~i ~j ~source ~e ~expr)
      ~a)))

(defmacro and-layout
  ([nav stor region i j idx expr]
   `(let [fd# (.fd ~stor)]
      (loop [~j 0]
        (if (< ~j fd#)
          (let [end# (.end ~nav ~region ~j)]
            (and (loop [~i (.start ~nav ~region ~j)]
                   (if (< ~i end#)
                     (let [~idx (.index ~stor ~i ~j)]
                       (and ~expr (recur (inc ~i))))
                     true))
                 (recur (inc ~j))))
          true))))
  ([nav stor region i j expr]
   `(let [fd# (.fd ~stor)]
      (loop [~j 0]
        (if (< ~j fd#)
          (let [end# (.end ~nav ~region ~j)]
            (and (loop [~i (.start nav# ~region ~j)]
                   (if (< ~i end#)
                     (and ~expr (recur (inc ~i)))
                     true))
                 (recur (inc ~j))))
          true))))
  ([a i j idx expr]
   `(let [nav# (navigator ~a)
          stor# (storage ~a)
          region# (region ~a)]
      (and-layout nav# stor# region# ~i ~j ~idx ~expr)))
  ([a i j expr]
   `(let [nav# (navigator ~a)
          stor# (storage ~a)
          region# (region ~a)]
      (and-layout nav# stor# region# ~i ~j ~expr))))

;; ========================= Default value ===============================

(deftype SYDefault []
  RealDefault
  (entry [_ stor da buf ofst i j]
    (.get ^RealBufferAccessor da buf (+ ofst (.index ^DenseStorage stor j i)))))

(deftype TRDefault []
  RealDefault
  (entry [_ _ _ _ _ _ _]
    0.0))

(deftype UnitTRDefault []
  RealDefault
  (entry [_ _ _ _ _ i j]
    (if (= i j) 1.0 0.0)))

(let [sy-default (SYDefault.)
      tr-default (TRDefault.)
      tr-unit-default (UnitTRDefault.)]

  (defn real-default [type unit]
    (case type
      :sy sy-default
      :tr (if unit tr-unit-default tr-default)
      nil)))
