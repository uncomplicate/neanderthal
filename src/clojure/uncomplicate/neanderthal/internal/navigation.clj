(ns uncomplicate.neanderthal.internal.navigation
  (:import [uncomplicate.neanderthal.internal.api RealOrderNavigator UploNavigator StripeNavigator]))

(deftype ColumnRealOrderNavigator []
  RealOrderNavigator
  (sd [_ m n]
    m)
  (fd [_ m n]
    n)
  (index [_ offset ld i j]
    (+ offset (* ld j) i))
  (get [_ a i j]
    (.entry a i j))
  (set [_ a i j val]
    (.set a i j val))
  (stripe [_ a j]
    (.col a j)))

(deftype RowRealOrderNavigator []
  RealOrderNavigator
  (sd [_ m n]
    n)
  (fd [_ m n]
    m)
  (index [_ offset ld i j]
    (+ offset (* ld i) j))
  (get [_ a i j]
    (.entry a j i))
  (set [_ a i j val]
    (.set a j i val))
  (stripe [_ a i]
    (.row a i)))

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
