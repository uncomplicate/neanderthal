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
            RealChangeable RealMatrix]))

(deftype ColumnRealOrderNavigator []
  RealOrderNavigator
  (sd [_ m n]
    m)
  (fd [_ m n]
    n)
  (index [_ offset ld i j]
    (+ offset (* ld j) i))
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
    i))

(deftype NonUnitTopNavigator []
  StripeNavigator
  (start [_ _ i]
    0)
  (end [_ _ i]
    (inc i)))

(deftype UnitBottomNavigator []
  StripeNavigator
  (start [_ _ i]
    (inc i))
  (end [_ n _]
    n))

(deftype NonUnitBottomNavigator []
  StripeNavigator
  (start [_ _ i]
    i)
  (end [_ n _]
    n))

(def non-unit-top-navigator (NonUnitTopNavigator.))
(def unit-top-navigator (UnitTopNavigator.))
(def non-unit-bottom-navigator (NonUnitBottomNavigator.))
(def unit-bottom-navigator (UnitBottomNavigator.))
