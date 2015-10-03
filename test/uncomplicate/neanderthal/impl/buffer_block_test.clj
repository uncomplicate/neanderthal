(ns uncomplicate.neanderthal.impl.buffer-block-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.clojurecl.core :refer [release]]
            [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.impl.buffer-block :refer :all])
  (:import [clojure.lang IFn$LLD IFn$LD IFn$DD]
           [uncomplicate.neanderthal.protocols RealBufferAccessor]
           [uncomplicate.neanderthal.impl.buffer_block
            RealBlockVector RealGeneralMatrix]))

(def accessorf double-accessor)

(deftype DummyFactory []
  Factory
  (create-vector1 [this n source]
    (if (and (<= (long n) (.count double-accessor source))
             (instance? java.nio.ByteBuffer source) )
      (->RealBlockVector this double-accessor nil (.entryType double-accessor)
                         true source n 1)
      (throw (IllegalArgumentException.
              (format "I can not create an %d element vector from %d-element %s."
                      n (.count double-accessor source) (class source))))))
  (data-accessor [_]
    double-accessor)
  (vector-engine [_ _ _ _ _]
    nil)
  (matrix-engine [_ _ _ _ _ _]
    nil))

(defn rbv ^RealBlockVector [^RealBufferAccessor access s ^long n ^long strd]
  (->RealBlockVector (->DummyFactory) access nil (.entryType access) true
                     (.toBuffer ^RealBufferAccessor accessorf s) n strd))

(defn rgm ^RealGeneralMatrix [^RealBufferAccessor access s m n ld ord]
  (->RealGeneralMatrix (->DummyFactory) access nil (.entryType access) true
                     (.toBuffer ^RealBufferAccessor accessorf s) m n ld ord))

(facts "Equality and hash code."
       (let [x1 (rbv accessorf [1 2 3 4] 4 1)
             y1 (rbv accessorf [1 2 3 4] 4 1)
             x2 (rbv accessorf [1 2 3 4] 2 2)
             y2 (rbv accessorf [1 3] 2 1)
             y3 (rbv accessorf [1 2 3 5] 4 1)]
         (.equals x1 nil) => false
         (= x1 y1) => true
         (= x1 y2) => false
         (= x2 y2) => true
         (= x1 y3) => false))

(facts "Group methods."
       (let [x (rbv accessorf [1 2 3] 3 1)
             y (rbv accessorf [2 3 4] 3 1)]

         (zero x) => (rbv accessorf [0 0 0] 3 1)
         (identical? x (zero x)) => false)

       (let [a (rgm accessorf [1 2 3 4 5 6] 2 3 2 DEFAULT_ORDER)
             ac (rgm accessorf [0 0 0 0 0 0] 2 3 2 DEFAULT_ORDER)]
         (zero a) => ac
         (identical? a (zero a)) => false))

(facts "IFn implementation for real block vector"
       (let [x (rbv accessorf [1 2 3 4] 4 1)]
         (x 2) => 3.0
         (x 5) => (throws IndexOutOfBoundsException)
         (x -1) => (throws IndexOutOfBoundsException)
         (instance? clojure.lang.IFn x) => true
         (.invokePrim ^IFn$LD x 0) => 1.0))

(let [fx (fn [] (rbv accessorf [1 2 3 4] 4 1))
      fy (fn [] (rbv accessorf [2 3 4 5 6] 5 1))
      x (fx)
      pf1 (fn ^double [^double x] (double (+ x 1.0)))
      pf2 (fn ^double [^double x ^double y] (double (+ x y)))
      pf3 (fn ^double [^double x ^double y ^double z] (double (+ x y z)))
      pf4 (fn ^double [^double x ^double y ^double z ^double w]
            (double (+ x y z w)))]
  (facts "Functor implementation for real block vector"
         (instance? IFn$DD pf1) => true

         (fmap! (fx) pf1) => (rbv accessorf [2 3 4 5] 4 1)
         (fmap! x pf1) => x

         (fmap! (fx) pf2 (fy)) => (rbv accessorf [3 5 7 9] 4 1)
         (fmap! x pf2 (fy)) => x

         (fmap! (fx) pf3 (fy) (fy)) => (rbv accessorf [5 8 11 14] 4 1)
         (fmap! x pf3 (fy) (fy)) => x

         (fmap! (fx) pf4 (fy) (fy) (fy)) => (rbv accessorf [7 11 15 19] 4 1)
         (fmap! x pf4 (fy) (fy) (fy)) => x

         (fmap! (fx) + (fy) (fy) (fy) [(fy)])
         => (throws UnsupportedOperationException)))

(let [x (rbv accessorf [1 2 3 4] 4 1)
      *' (fn ^double [^double x ^double y]
           (double (* x y)))
      +' (fn ^double [^double x ^double y]
           (double (+ x y)))]
  (facts "Fold implementation for double vector"

         (fold x) => 10.0
         (fold x *' 1.0) => 24.0
         (fold x +' 0.0) => (fold x)))

(let [y (rbv accessorf [2 3 4 5 6] 5 1)
      x (rbv accessorf [1 2 3 4] 4 1)
      pf1 (fn ^double [^double res ^double x] (+ x res))
      pf1o (fn [res ^double x] (conj res x))
      pf2 (fn ^double [^double res ^double x ^double y] (+ res x y))
      pf2o (fn [res ^double x ^double y] (conj res [x y]))
      pf3 (fn ^double [^double res ^double x ^double y ^double z]
            (+ res x y z))
      pf3o (fn [res ^double x ^double y ^double z] (conj res [x y z]))]
  (facts "Reducible implementation for double vector"

         (freduce x 1.0 pf1) => 11.0
         (freduce x [] pf1o) => [1.0 2.0 3.0 4.0]

         (freduce x 1.0 pf2 y) => 25.0
         (freduce x [] pf2o y)
         => [[1.0 2.0] [2.0 3.0] [3.0 4.0] [4.0 5.0]]

         (freduce x 1.0 pf3 y y) => 39.0
         (freduce x [] pf3o y y)
         => [[1.0 2.0 2.0] [2.0 3.0 3.0] [3.0 4.0 4.0] [4.0 5.0 5.0]]

         (freduce x 1.0 + y y [y])
         => (throws UnsupportedOperationException)))

(facts "Vector as a sequence"
       (seq (rbv accessorf [1 2 3] 3 1)) => '(1.0 2.0 3.0)
       (seq (rbv accessorf [1 0 2 0 3 0] 3 2)) => '(1.0 2.0 3.0))

(facts "Matrix methods."
       (let [a (rgm accessorf [1 2 3 4 5 6] 2 3 2 DEFAULT_ORDER)
             ac (rgm accessorf [0 0 0 0 0 0] 2 3 2 DEFAULT_ORDER)]
         (.mrows a) => 2
         (.ncols a) => 3

         (.row a 1) => (rbv accessorf [2 0 4 0 6 0] 3 2)

         (.col a 1) => (rbv accessorf [3 4] 2 1)))

(facts "IFn implementation for double general matrix"
       (let [x (rgm accessorf [1 2 3 4 5 6] 2 3 2 DEFAULT_ORDER)]
         (x 1 2) => 6.0
         (x 2 1) => (throws IndexOutOfBoundsException)
         (x -1 3) => (throws IndexOutOfBoundsException)
         (instance? clojure.lang.IFn x) => true
         (.invokePrim x 0 0) => 1.0))

(let [fx (fn [] (rgm accessorf [1 2 3 4 5 6] 2 3 2 DEFAULT_ORDER))
      fy (fn [] (rgm accessorf [2 3 4 5 6 7] 2 3 2 DEFAULT_ORDER))
      x (fx)
      pf1 (fn ^double [^double x] (double (+ x 1.0)))
      pf2 (fn ^double [^double x ^double y] (double (+ x y)))
      pf3 (fn ^double [^double x ^double y ^double z] (double (+ x y z)))
      pf4 (fn ^double [^double x ^double y ^double z ^double w]
            (double (+ x y z w)))]
  (facts "Functor implementation for double general matrix"
         (instance? clojure.lang.IFn$DD pf1) => true

         (fmap! (fx) pf1) => (fy)
         (fmap! x pf1) => x

         (fmap! (fx) pf2 (fy)) => (rgm accessorf [3 5 7 9 11 13] 2 3 2 DEFAULT_ORDER)
         (fmap! x pf2 (fy)) => x

         (fmap! (fx) pf3 (fy) (fy)) => (rgm accessorf [5 8 11 14 17 20] 2 3 2 DEFAULT_ORDER)
         (fmap! x pf3 (fy) (fy)) => x

         (fmap! (fx) pf4 (fy) (fy) (fy))
         => (rgm accessorf [7 11 15 19 23 27] 2 3 2 DEFAULT_ORDER)
         (fmap! x pf4 (fy) (fy) (fy)) => x

         (fmap! (fx) + (fy) (fy) (fy) [(fy)])
         => (throws UnsupportedOperationException)))

(let [x (rgm accessorf [1 2 3 4 5 6] 2 3 2 DEFAULT_ORDER)
      *' (fn ^double [^double x ^double y] (double (* x y)))
      +' (fn ^double [^double x ^double y] (double (+ x y)))]
  (facts "Fold implementation for real matrix"
         (fold x) => 21.0
         (fold x *' 1.0) => 720.0
         (fold x +' 0.0) => (fold x)))

(let [x (rgm accessorf [1 2 3 4 5 6] 2 3 2 DEFAULT_ORDER)
      y (rgm accessorf [2 3 4 5 6 7] 2 3 2 DEFAULT_ORDER)
      pf1 (fn ^double [^double res ^double x] (+ x res))
      pf1o (fn [res ^double x] (conj res x))
      pf2 (fn ^double [^double res ^double x ^double y] (+ res x y))
      pf2o (fn [res ^double x ^double y] (conj res [x y]))
      pf3 (fn ^double [^double res ^double x ^double y ^double z]
            (+ res x y z))
      pf3o (fn [res ^double x ^double y ^double z]
             (conj res [x y z]))]
  (facts "Reducible implementation for real matrix"

         (freduce x 1.0 pf1) => 22.0
         (freduce x [] pf1o) => [1.0 2.0 3.0 4.0 5.0 6.0]

         (freduce x 1.0 pf2 y) => 49.0
         (freduce x [] pf2o y)
         => [[1.0 2.0] [2.0 3.0] [3.0 4.0] [4.0 5.0] [5.0 6.0] [6.0 7.0]]

         (freduce x 1.0 pf3 y y) => 76.0
         (freduce x [] pf3o y y)
         => [[1.0 2.0 2.0] [2.0 3.0 3.0] [3.0 4.0 4.0] [4.0 5.0 5.0]
             [5.0 6.0 6.0] [6.0 7.0 7.0]]

         (freduce x 1.0 + y y [y])
         => (throws UnsupportedOperationException)))

(let [a (rgm accessorf [1 2 3 4 5 6] 2 3 4 DEFAULT_ORDER)
      col-a (.col a 0)
      sub-a (.submatrix a 0 0 1 1)]
  (facts "RealBlockVector and RealBlockMatrix release."
         (release col-a) => true
         (release col-a) => true
         (release sub-a) => true
         (release sub-a) => true
         (release a) => true
         (release a) => true))
