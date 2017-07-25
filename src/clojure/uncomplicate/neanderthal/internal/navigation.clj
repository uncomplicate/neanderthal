;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.navigation
  (:import [clojure.lang IFn$LLDD]
           [uncomplicate.neanderthal.internal.api RealOrderNavigator UploNavigator StripeNavigator
            BandNavigator RealChangeable RealMatrix]))

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
    (.index this offset ld kl ku (max 0 (- k)) (max 0 k)))
  (stripeIndex [_ offset ld kl ku i j]
    (+ offset (* j ld) (- ku j) i)))

(deftype RowBandNavigator []
  BandNavigator
  (height [_ m n kl _]
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
  (index [this offset ld kl ku k]
    (.index this offset ld kl ku (max 0 (- k)) (max 0 k)))
  (stripeIndex [_ offset ld kl ku i j]
    (+ offset (* i ld) (- kl i) j)))

(def col-band-navigator (ColumnBandNavigator.))
(def row-band-navigator (RowBandNavigator.))
