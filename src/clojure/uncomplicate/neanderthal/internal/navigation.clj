;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.navigation
  (:require [uncomplicate.neanderthal.internal
             [api :refer :all]
             [common :refer [dragan-says-ex]]])
  (:import [clojure.lang IFn$LLDD]
           [uncomplicate.neanderthal.internal.api LayoutNavigator RealLayoutNavigator DenseStorage
            FullStorage Region RealDefault RealBufferAccessor RealChangeable
            RealMatrix Matrix]))

;; ======================== Region  =================================================

(deftype BandRegion [^long m ^long n ^long kl ^long ku ^long uplo ^long diag]
  Object
  (hashCode [a]
    (-> (hash-combine :BandRegion) (hash-combine m) (hash-combine n) (hash-combine kl) (hash-combine ku)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? BandRegion b)
             (= m (.m ^BandRegion b)) (= n (.n ^BandRegion b))
             (= kl (.kl ^BandRegion b)) (= ku (.ku ^BandRegion b)))))
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

(defn band-region
  ([^long m ^long n ^long kl ^long ku]
   (BandRegion. m n kl ku (cond (< 0 kl) 122 (< 0 ku) 121 :default 0)
                (if (or (= -1 kl) (= -1 ku)) 132 131)))
  ([^long n lower? diag-unit?]
   (let [diag-pad (if diag-unit? -1 0)
         diag (if diag-unit? 132 131)]
     (if lower?
       (band-region n n (max 0 (dec n)) diag-pad)
       (band-region n n diag-pad (max 0 (dec n))))))
  ([^long n lower?]
   (band-region n lower? false)))

(deftype GERegion [^long m ^long n]
  Object
  (hashCode [a]
    (-> (hash-combine :GERegion) (hash-combine m) (hash-combine n)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? GERegion b)
             (= m (.m ^GERegion b)) (= n (.n ^GERegion b)))))
  (toString [a]
    (format "#GERegion[mxn:%dx%d]" m n))
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

(declare real-column-navigator)
(declare real-row-navigator)

(deftype RealColumnNavigator []
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
  RealLayoutNavigator
  (get [_ a i j]
    (.entry ^RealMatrix a i j))
  (set [_ a i j val]
    (.set ^RealChangeable a i j val))
  (invokePrimitive [_ f i j val]
    (.invokePrim ^IFn$LLDD f i j val))
  Flippable
  (flip [_]
    real-row-navigator))

(deftype RealRowNavigator []
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
  RealLayoutNavigator
  (get [_ a i j]
    (.entry ^RealMatrix a j i))
  (set [_ a i j val]
    (.set ^RealChangeable a j i val))
  (invokePrimitive [_ f i j val]
    (.invokePrim ^IFn$LLDD f j i val))
  Flippable
  (flip [_]
    real-column-navigator))

(def ^:const real-column-navigator (RealColumnNavigator.))
(def ^:const real-row-navigator (RealRowNavigator.))

(defn layout-navigator [column?]
  (if column? real-column-navigator real-row-navigator))

(defn real-navigator ^RealLayoutNavigator [a]
  (navigator a))

;; =================== Full Storage ========================================

(deftype StripeFullStorage [^long sd ^long fd ^long ld]
  Object
  (hashCode [a]
    (-> (hash-combine :FullStorage) (hash-combine sd) (hash-combine fd) (hash-combine ld)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? StripeFullStorage b) (= sd (.sd ^StripeFullStorage b))
             (= fd (.fd ^StripeFullStorage b)) (= ld (.ld ^StripeFullStorage b)))))
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
    (* ld fd)))

(defn full-storage
  ([column? ^long m ^long n ^long ld]
   (if column?
     (StripeFullStorage. m n (max m ld))
     (StripeFullStorage. n m (max n ld))))
  ([column? ^long m ^long n]
   (if column?
     (StripeFullStorage. m n m)
     (StripeFullStorage. n m n)))
  (^FullStorage [a]
   (storage a)))

;; =================== Full Band Storage =========================================

(deftype BandStorage [^long h ^long w ^long ld ^long kl ^long ku ^long sym-k]
  Object
  (hashCode [a]
    (-> (hash-combine :BandStorage) (hash-combine ld) (hash-combine kl) (hash-combine ku)))
  (equals [a b]
    (or (identical? a b)
        (and (instance? BandStorage b) (= h (.h ^BandStorage b))
             (= w (.w ^BandStorage b)) (= ld (.ld ^BandStorage b))
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
    (* ld w)))

(defn band-storage
  ([column? m n ld kl ku]
   (let [m (long m)
         n (long n)
         kl (long kl)
         ku (long ku)
         h (inc (+ kl ku))
         ld (max h (long ld))
         sym-k (cond (= 0 kl) ku (= 0 ku) kl :default 0)]
     (if column?
       (BandStorage. h (min n (+ (min m n) ku)) ld kl ku sym-k)
       (BandStorage. h (min m (+ (min m n) kl)) ld ku kl sym-k))))
  ([column? m n kl ku]
   (band-storage column? m n (if column? m n) kl ku))
  ([column? ^long n lower? diag-unit?]
   (let [unit-pad (if diag-unit? -1 0)]
     (if lower?
       (band-storage column? n n (dec n) unit-pad)
       (band-storage column? n n unit-pad (dec n))))))

;; =================== Packed Storage ======================================

(deftype TopPackedStorage [^long n]
  Object
  (hashCode [a]
    (-> (hash-combine :TopPackedStorage) (hash-combine n)))
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
  Object
  (hashCode [a]
    (-> (hash-combine :BottomPackedStorage) (hash-combine n)))
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

(defn packed-storage [column? lower? ^long n]
  (if column?
    (if lower?
      (BottomPackedStorage. n)
      (TopPackedStorage. n))
    (if lower?
      (TopPackedStorage. n)
      (BottomPackedStorage. n))))

;; ========================= Default value ===============================

(deftype SYDefault []
  RealDefault
  (entry [_ stor da buf ofst i j]
    (.get ^RealBufferAccessor da buf (+ ofst (.index ^DenseStorage stor j i)))))

(deftype SBDefault []
  RealDefault
  (entry [_ stor da buf ofst i j]
    (let [k (.sym-k ^BandStorage stor)]
      (if (<= (- j k) i (+ j k))
        (.get ^RealBufferAccessor da buf (+ ofst (.index ^BandStorage stor j i)))
        0.0))))

(deftype ZeroDefault []
  RealDefault
  (entry [_ _ _ _ _ _ _]
    0.0))

(deftype UnitDefault []
  RealDefault
  (entry [_ _ _ _ _ i j]
    (if (= i j) 1.0 0.0)))

(def sy-default (SYDefault.))
(def sb-default (SBDefault.))
(def zero-default (ZeroDefault.))
(def unit-default (UnitDefault.))

(defn real-default
  ([type diag-unit]
   (case type
     :sy sy-default
     :tr (if diag-unit unit-default zero-default)
     :sb sb-default
     :tb (if diag-unit unit-default zero-default)
     :sp sy-default
     :tp (if diag-unit unit-default zero-default)
     (dragan-says-ex "Unknown default. This part should not depend on your code. Please send me a bug report." {:type type})))
  ([type]
   (case type
     :sy sy-default
     :tr zero-default
     :sb sb-default
     :tb zero-default
     :sp sy-default
     :tp zero-default
     (dragan-says-ex "Unknown default. This part should not depend on your code. Please send me a bug report." {:type type}))))

;; ======================= Layout utillities ==============================

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
