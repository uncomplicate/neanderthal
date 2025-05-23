;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.navigation
  (:refer-clojure :exclude [abs])
  (:require [uncomplicate.commons
             [core :refer [Info info]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal.internal.api :refer :all])
  (:import [clojure.lang IFn$LLDD]
           [uncomplicate.neanderthal.internal.api LayoutNavigator DenseStorage FullStorage Region
            Default RealAccessor IntegerAccessor RealChangeable IntegerChangeable
            RealMatrix IntegerMatrix Matrix RealLayoutFlipper IntegerLayoutFlipper]))

;; ======================== Region  =================================================

(defn diag-unit? [^Region reg]
  (.isDiagUnit reg))

(defn lower? [^Region reg]
  (.isLower reg))

(defn upper? [^Region reg]
  (.isUpper reg))

(defn uplo= [^Region reg1 ^Region reg2]
  (= (.uplo reg1) (.uplo reg2)))

(deftype BandRegion [^long m ^long n ^long kl ^long ku ^long uplo ^long diag]
  Object
  (hashCode [a]
    (-> (hash :BandRegion) (hash-combine m) (hash-combine n) (hash-combine kl) (hash-combine ku)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? BandRegion b)
             (= m (.m ^BandRegion b)) (= n (.n ^BandRegion b))
             (= kl (.kl ^BandRegion b)) (= ku (.ku ^BandRegion b)))))
  (toString [a]
    (format "#BandRegion[mxn:%dx%d, kl:%d, ku:%d]" m n kl ku))
  Info
  (info [r]
    {:region-type :band
     :m m
     :n n
     :kl kl
     :ku ku
     :surface (.surface r)
     :uplo (dec-property (.uplo r))
     :diag (if (.isDiagUnit r) :unit :non-unit)})
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
    (= 122 uplo))
  (isUpper [_]
    (= 121 uplo))
  (isDiagUnit [_]
    (or (= -1 kl) (= -1 ku)))
  (uplo [_]
    uplo)
  (diag [_]
    diag)
  (surface [_]
    (- (* m n)
       (unchecked-divide-int (* (- m kl) (dec (- m kl))) 2)
       (unchecked-divide-int (* (- n ku) (dec (- n ku))) 2)))
  (kl [_]
    kl)
  (ku [_]
    ku)
  Flippable
  (flip [_]
    (BandRegion. n m ku kl (case uplo 121 122 122 121 0) diag)))

(defn square-band-region
  [^long n ^long kl ^long ku ^Boolean lower?]
  (let [kl (min (max 0 (dec n)) kl)
        ku (min (max 0 (dec n)) ku)]
    (BandRegion. n n kl ku (if-not lower? 121 122)
                 (if (or (= -1 kl) (= -1 ku)) 132 131))))

(defn band-region
  ([^long m ^long n ^long kl ^long ku]
   (let [kl (min (max 0 (dec m)) kl)
         ku (min (max 0 (dec n)) ku)]
     (BandRegion. m n kl ku (cond (< 0 kl) 122 (< 0 ku) 121 :default 122)
                  (if (or (= -1 kl) (= -1 ku)) 132 131))))
  ([^long n ^Boolean lower? ^Boolean diag-unit?]
   (let [diag-pad (if diag-unit? -1 0)]
     (if lower?
       (square-band-region n (max 0 (dec n)) diag-pad lower?)
       (square-band-region n diag-pad (max 0 (dec n)) lower?))))
  ([^long n lower?]
   (band-region n lower? false)))

(defn tb-region
  ([^long n ^long k lower? diag-unit?]
   (let [diag-pad (if diag-unit? -1 0)]
     (if lower?
       (square-band-region n (min (max 0 k) (max 0 (dec n))) diag-pad lower?)
       (square-band-region n diag-pad (min (max 0 k) (max 0 (dec n))) lower?)))))

(defn sb-region
  ([^long n ^long k lower?]
   (if lower?
     (square-band-region n (min (max 0 k) (max 0 (dec n))) 0 lower?)
     (square-band-region n 0 (min (max 0 k) (max 0 (dec n))) lower?))))

(deftype GERegion [^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash :GERegion) (hash-combine m) (hash-combine n)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? GERegion b)
             (= m (.m ^GERegion b)) (= n (.n ^GERegion b)))))
  (toString [a]
    (format "#GERegion[mxn:%dx%d]" m n))
  Info
  (info [r]
    {:region-type :full
     :m m
     :n n
     :kl (.kl r)
     :ku (.ku r)
     :surface (.surface r)
     :uplo (dec-property (.uplo r))
     :diag (if (.isDiagUnit r) :unit :non-unit)})
  Region
  (accessible [_ _ _]
    true)
  (colStart [_ _]
    0)
  (colEnd [_ _]
    m)
  (rowStart [_ _]
    0)
  (rowEnd [_ _]
    n)
  (isLower [_]
    false)
  (isUpper [_]
    false)
  (isDiagUnit [_]
    false)
  (uplo [_]
    122)
  (diag [_]
    131)
  (surface [_]
    (* m n))
  (kl [_]
    (dec m))
  (ku [_]
    (dec n))
  Flippable
  (flip [_]
    (GERegion. n m)))

(defn ge-region [^long m ^long n]
  (GERegion. m n))

;; =================== Navigator ==========================================

(deftype RealNoFlip []
  RealLayoutFlipper
  (get [_ a i j]
    (.entry ^RealMatrix a i j))
  (set [_ a i j val]
    (.set ^RealChangeable a i j val)))

(deftype RealFlip []
  RealLayoutFlipper
  (get [_ a i j]
    (.entry ^RealMatrix a j i))
  (set [_ a i j val]
    (.set ^RealChangeable a j i val)))

(def ^:private real-no-flip (->RealNoFlip))
(def ^:private real-flip (->RealFlip))

(deftype IntegerNoFlip []
  IntegerLayoutFlipper
  (get [_ a i j]
    (.entry ^IntegerMatrix a i j))
  (set [_ a i j val]
    (.set ^IntegerChangeable a i j val)))

(deftype IntegerFlip []
  IntegerLayoutFlipper
  (get [_ a i j]
    (.entry ^IntegerMatrix a j i))
  (set [_ a i j val]
    (.set ^IntegerChangeable a j i val)))

(def ^:private integer-no-flip (->IntegerNoFlip))
(def ^:private integer-flip (->IntegerFlip))

(def ^:private real-column-navigator (volatile! nil))
(def ^:private real-row-navigator (volatile! nil))

(deftype ColumnNavigator []
  Info
  (info [n]
    {:layout :column})
  LayoutNavigator
  (start [_ region j]
    (.colStart ^Region region j))
  (end [_ region j]
    (.colEnd ^Region region j))
  (stripe [_ a j]
    (.col ^Matrix a j))
  (index [_ stor i j]
    (.index ^DenseStorage stor i j))
  (isColumnMajor [_]
    true)
  (isRowMajor [_]
    false)
  (layout [_]
    102)
  Flippable
  (flip [_]
    @real-row-navigator)
  Flipper
  (real-flipper [_]
    real-no-flip)
  (integer-flipper [_]
    integer-no-flip))

(deftype RowNavigator []
  Info
  (info [n]
    {:layout :row})
  LayoutNavigator
  (start [_ region j]
    (.rowStart ^Region region j))
  (end [_ region j]
    (.rowEnd ^Region region j))
  (stripe [_ a j]
    (.row ^Matrix a j))
  (index [_ stor i j]
    (.index ^DenseStorage stor j i))
  (isColumnMajor [_]
    false)
  (isRowMajor [_]
    true)
  (layout [_]
    101)
  Flippable
  (flip [_]
    @real-column-navigator)
  Flipper
  (real-flipper [_]
    real-flip)
  (integer-flipper [_]
    integer-flip))

(deftype DiagonalNavigator []
  Info
  (info [n]
    {:layout :diagonal})
  LayoutNavigator
  (start [_ _ _]
    0)
  (end [_ region j]
    (if (= 0 j) (.surface region) 0))
  (stripe [_ a j]
    (.dia ^Matrix a j))
  (index [_ stor i j]
    (.index ^DenseStorage stor i (- j i)))
  (isColumnMajor [_]
    false)
  (isRowMajor [_]
    false)
  (layout [_]
    -1)
  Flippable
  (flip [this]
    this)
  Flipper
  (real-flipper [_]
    real-no-flip)
  (integer-flipper [_]
    integer-no-flip))

(vreset! real-column-navigator (->ColumnNavigator))
(vreset! real-row-navigator (->RowNavigator))

(def diagonal-navigator (DiagonalNavigator.))

(defn layout-navigator [^Boolean column?]
  (if column? @real-column-navigator @real-row-navigator))

;; =================== Full Storage ========================================

(deftype StripeFullStorage [^long sd ^long fd ^long ld]
  Info
  (info [s]
    {:storage-type :full
     :sd sd
     :ld ld
     :fd fd
     :gapless (.isGapless s)
     :capacity (.capacity s)})
  Object
  (hashCode [a]
    (-> (hash :FullStorage) (hash-combine sd) (hash-combine fd) (hash-combine ld)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? StripeFullStorage b) (= sd (.sd ^StripeFullStorage b))
             (= fd (.fd ^StripeFullStorage b)))))
  (toString [a]
    (format "#FullStorage[sd:%d, fd:%d, ld:%d]" sd fd ld))
  FullStorage
  (sd [_]
    sd)
  (ld [_]
    ld)
  DenseStorage
  (index [_ i j]
    (+ (* j ld) i))
  (fd [_]
    fd)
  (isGapless [_]
    (= sd ld))
  (capacity [_]
    (+ (* ld (max 0 (dec fd))) sd)))

(defn full-storage
  (^FullStorage [^Boolean column? ^long m ^long n ^long ld]
   (if column?
     (StripeFullStorage. m n (max m ld))
     (StripeFullStorage. n m (max n ld))))
  (^FullStorage [^Boolean column? ^long m ^long n]
   (if column?
     (StripeFullStorage. m n m)
     (StripeFullStorage. n m n)))
  (^FullStorage [a]
   (storage a)))

;; =================== Full Band Storage =========================================

(deftype BandStorage [^long h ^long w ^long ld ^long kl ^long ku]
  Info
  (info [s]
    {:storage-type :band
     :height h
     :width w
     :kl kl
     :ku ku
     :sd h
     :ld ld
     :fd w
     :gapless (.isGapless s)
     :capacity (.capacity s)})
  Object
  (hashCode [a]
    (-> (hash :BandStorage) (hash-combine ld) (hash-combine kl) (hash-combine ku)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? BandStorage b) (= h (.h ^BandStorage b))
             (= w (.w ^BandStorage b))
             (= kl (.kl ^BandStorage b)) (= ku (.ku ^BandStorage b)))))
  (toString [a]
    (format "#BandStorage[h:%d, w:%d, ld:%d, kl:%d, ku:%d]" h w ld kl ku))
  FullStorage
  (sd [_]
    h)
  (ld [_]
    ld)
  DenseStorage
  (index [_ i j]
    (+ (* j ld) (- ku j) i))
  (fd [_]
    w)
  (isGapless [_]
    (= 0 kl ku))
  (capacity [_]
    (+ (* ld (max 0 (dec w))) h)))

(defn band-storage
  ([^Boolean column? m n ld kl ku]
   (let [m (long m)
         n (long n)
         kl (max 0 (long kl))
         ku (max 0 (long ku))
         h (inc (+ kl ku))
         ld (max h (long ld))]
     (if column?
       (BandStorage. h (min n (+ (min m n) ku)) ld kl ku)
       (BandStorage. h (min m (+ (min m n) kl)) ld ku kl))))
  ([^Boolean column? m n kl ku]
   (band-storage column? m n (inc (+ (long kl) (long ku))) kl ku)))

(defn uplo-storage [^Boolean column? ^long n ^long k lower?]
  (if lower?
    (band-storage column? n n (min k (dec n)) 0)
    (band-storage column? n n 0 (min k (dec n)))))

;; =================== Packed Storage ======================================

(deftype TopPackedStorage [^long n]
  Info
  (info [s]
    {:storage-type :packed
     :triangle :top
     :n n
     :fd n
     :gapless true
     :capacity (.capacity s)})
  Object
  (hashCode [a]
    (-> (hash :TopPackedStorage) (hash-combine n)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? TopPackedStorage b) (= n (.n ^TopPackedStorage b)))))
  (toString [a]
    (format "#TopPackedstorage[n:%d]" n))
  DenseStorage
  (index [_ i j]
    (+ i (unchecked-divide-int (* j (inc j)) 2)))
  (fd [_]
    n)
  (isGapless [_]
    true)
  (capacity [_]
    (unchecked-divide-int (* n (inc n)) 2)))

(deftype BottomPackedStorage [^long n]
  Info
  (info [s]
    {:storage-type :packed
     :triangle :bottom
     :n n
     :fd n
     :gapless true
     :capacity (.capacity s)})
  Object
  (hashCode [a]
    (-> (hash :BottomPackedStorage) (hash-combine n)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? BottomPackedStorage b) (= n (.n ^BottomPackedStorage b)))))
  (toString [a]
    (format "#BottomPackedstorage[n:%d]" n))
  DenseStorage
  (index [_ i j]
    (+ i (unchecked-divide-int (* j (dec (- (* 2 n) j))) 2)))
  (fd [_]
    n)
  (isGapless [_]
    true)
  (capacity [_]
    (unchecked-divide-int (* n (inc n)) 2)))

(defn packed-storage [^Boolean column? ^Boolean lower? ^long n]
  (if column?
    (if lower?
      (BottomPackedStorage. n)
      (TopPackedStorage. n))
    (if lower?
      (TopPackedStorage. n)
      (BottomPackedStorage. n))))

;; ========================= Diagonal Storage ============================

(deftype TridiagonalStorage [^long n ^long cap]
  Info
  (info [s]
    {:storage-type :tridiagonal
     :n n
     :capacity (.capacity s)})
  Object
  (hashCode [a]
    (-> (hash :TridiagonalStorage) (hash-combine n)))
  (equals [a b]
    (or (identical? a b) (and (instance? TridiagonalStorage b) (= n (.n ^TridiagonalStorage b)))))
  (toString [a]
    (format "#TridiagonalStorage[n:%d]" n))
  DenseStorage
  (index [_ i j]
    (case j
      0 i
      1 (+ n i)
      (+ n (max 0 (dec n)) i)))
  (fd [_]
    1)
  (isGapless [_]
    true)
  (capacity [_]
    cap))

(deftype BidiagonalStorage [^long n ^long cap]
  Info
  (info [s]
    {:storage-type :bidiagonal
     :n n
     :capacity (.capacity s)})
  Object
  (hashCode [a]
    (-> (hash :BidiagonalStorage) (hash-combine n)))
  (equals [a b]
    (or (identical? a b) (and (instance? BidiagonalStorage b) (= n (.n ^BidiagonalStorage b)))))
  (toString [a]
    (format "#BidiagonalStorage[n:%d]" n))
  DenseStorage
  (index [_ i j]
    (+ (* n j) i))
  (fd [_]
    1)
  (isGapless [_]
    true)
  (capacity [_]
    cap))

(defn diagonal-storage [^long n matrix-type]
  (case matrix-type
    :gd (BidiagonalStorage. n n)
    :gt (TridiagonalStorage. n (+ n (* 2 (max 0 (dec n))) (max 0 (- n 2))))
    :dt (TridiagonalStorage. n (+ n (* 2 (max 0 (dec n)))))
    :st (BidiagonalStorage. n (+ n (max 0 (dec n))))
    (dragan-says-ex "Unknown (tri)diagonal matrix type." {:type type})))

;; ========================= Default value ===============================

(deftype SYDefault []
  Default
  (realEntry [_ nav stor da buf ofst i j]
    (.get ^RealAccessor da buf (+ ofst (.index ^LayoutNavigator nav ^DenseStorage stor j i))))
  (integerEntry [_ nav stor da buf ofst i j]
    (.get ^IntegerAccessor da buf (+ ofst (.index ^LayoutNavigator nav ^DenseStorage stor j i)))))

(deftype SBDefault []
  Default
  (realEntry [_ nav stor da buf ofst i j]
    (let [k (+ (.kl ^BandStorage stor) (.ku ^BandStorage stor))]
      (if (<= (- j k) i (+ j k))
        (.get ^RealAccessor da buf (+ ofst (.index ^LayoutNavigator nav ^BandStorage stor j i)))
        0.0)))
  (integerEntry [_ nav stor da buf ofst i j]
    (let [k (+ (.kl ^BandStorage stor) (.ku ^BandStorage stor))]
      (if (<= (- j k) i (+ j k))
        (.get ^IntegerAccessor da buf (+ ofst (.index ^LayoutNavigator nav ^BandStorage stor j i)))
        0))))

(deftype STDefault []
  Default
  (realEntry [_ nav stor da buf ofst i j]
    (if (< -2 (- i j) 2)
      (.get ^RealAccessor da buf (+ ofst (.index ^DenseStorage stor j (Math/abs (- i j)))))
      0.0))
  (integerEntry [_ nav stor da buf ofst i j]
    (if (< -2 (- i j) 2)
      (.get ^IntegerAccessor da buf (+ ofst (.index ^DenseStorage stor j (Math/abs (- i j)))))
      0)))

(deftype ZeroDefault []
  Default
  (realEntry [_ _ _ _ _ _ _ _]
    0.0)
  (integerEntry [_ _ _ _ _ _ _ _]
    0))

(deftype UnitDefault []
  Default
  (realEntry [_ _ _ _ _ _ i j]
    (if (= i j) 1.0 0.0))
  (integerEntry [_ _ _ _ _ _ i j]
    (if (= i j) 1 0)))

(def ^:const sy-default (SYDefault.))
(def ^:const sb-default (SBDefault.))
(def ^:const st-default (STDefault.))
(def ^:const zero-default (ZeroDefault.))
(def ^:const unit-default (UnitDefault.))

(defn default
  ([type diag-unit]
   (case type
     :sy sy-default
     :tr (if diag-unit unit-default zero-default)
     :gb zero-default
     :sb sb-default
     :tb (if diag-unit unit-default zero-default)
     :sp sy-default
     :tp (if diag-unit unit-default zero-default)
     :td zero-default
     :gt zero-default
     :gd zero-default
     :dt zero-default
     :st st-default
     (dragan-says-ex "Unknown default. This part should not depend on your code. Please send me a bug report."
                     {:type type})))
  ([type]
   (case type
     :sy sy-default
     :tr zero-default
     :gb zero-default
     :sb sb-default
     :tb zero-default
     :sp sy-default
     :tp zero-default
     :td zero-default
     :gt zero-default
     :gd zero-default
     :dt zero-default
     :st st-default
     (dragan-says-ex "Unknown default. This part should not depend on your code. Please send me a bug report."
                     {:type type}))))

;; ======================= Layout utilities ==============================

(defmacro doall-layout
  ([nav stor region i j idx cnt expr]
   `(let [fd# (.fd ~stor)]
      (loop [~j 0 cnt# 0]
        (when (< ~j fd#)
          (let [end# (.end ~nav ~region ~j)]
            (recur (inc ~j)
                   (long (loop [~i (.start ~nav ~region ~j) ~cnt cnt#]
                           (if (< ~i end#)
                             (let [~idx (.index ~stor ~i ~j)]
                               ~expr
                               (recur (inc ~i) (inc ~cnt)))
                             ~cnt)))))))))
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
                       (let [~idx (.index ~stor ~i ~j)
                             ~e (first src#)]
                         ~expr
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
                       (let [~e (first src#)]
                         ~expr
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

(defmacro dostripe-layout
  ([nav stor region len idx expr]
   `(dotimes [j# (.fd ~stor)]
      (let [start# (.start ~nav ~region j#)
            ~len (- (.end ~nav ~region j#) start#)
            ~idx (.index ~stor start# j#)]
        ~expr)))
  ([a len idx expr]
   `(let [nav# (navigator ~a)
          stor# (storage ~a)
          region# (region ~a)]
      (dostripe-layout nav# stor# region# ~len ~idx ~expr)
      ~a)))

(defmacro accu-layout
  ([nav stor region len idx acc init expr]
   `(let [n# (.fd ~stor)]
      (loop [j# 0 ~acc ~init]
        (if (< j# n#)
          (recur (inc j#)
                 (let [start# (.start ~nav ~region j#)
                       ~len (- (.end ~nav ~region j#) start#)
                       ~idx (.index ~stor start# j#)]
                   ~expr))
          ~acc))))
  ([a len idx acc init expr]
   `(let [nav# (navigator ~a)
          stor# (storage ~a)
          region# (region ~a)]
      (accu-layout nav# stor# region# ~len ~idx ~acc ~init ~expr))))
