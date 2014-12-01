(ns uncomplicate.neanderthal.cblas-test
  (:require [midje.sweet :refer :all]
            [vertigo
             [core :refer [wrap]]
             [structs :refer [float64]]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [cblas :refer :all]
             [real :refer [seq-to-buffer]]])
  (:import [uncomplicate.neanderthal.protocols
            RealVector RealMatrix]))

(set! *warn-on-reflection* false)

(facts "Equality and hash code."
       (let [x1 (->DoubleBlockVector (seq-to-buffer [1 2 3 4]) 4 1)
             y1 (->DoubleBlockVector (seq-to-buffer [1 2 3 4]) 4 1)
             x2 (->DoubleBlockVector (seq-to-buffer [1 2 3 4]) 2 2)
             y2 (->DoubleBlockVector (seq-to-buffer [1 3]) 2 1)
             y3 (->DoubleBlockVector (seq-to-buffer [1 2 3 5]) 4 1)]
         (.equals x1 nil) => false
         (= x1 y1) => true
         (= x1 y2) => false
         (= x2 y2) => true
         (= x1 y3) => false))

(facts "Carrier methods."
       (let [x (->DoubleBlockVector (seq-to-buffer [1 2 3 4]) 4 1)]
         (zero x) => (->DoubleBlockVector (seq-to-buffer [0 0 0 0]) 4 1)
         (identical? x (zero x)) => false))

(facts "IFn implementation for double vector"
       (let [x (->DoubleBlockVector (seq-to-buffer [1 2 3 4]) 4 1)]
         (x 2) => 3.0
         (x 5) => (throws IndexOutOfBoundsException)
         (x -1) => (throws IndexOutOfBoundsException)
         (instance? clojure.lang.IFn x) => true
         (.invokePrim x 0) => 1.0))

(let [fx (fn [] (->DoubleBlockVector (seq-to-buffer [1 2 3 4]) 4 1))
      fy (fn [] (->DoubleBlockVector (seq-to-buffer [2 3 4 5 6]) 5 1))
      x (fx)
      pf1 (fn ^double [^double x]
            (double (+ x 1.0)))
      pf2 (fn ^double [^double x ^double y]
            (double (+ x y)))
      pf3 (fn ^double [^double x ^double y ^double z]
            (double (+ x y z)))
      pf4 (fn ^double [^double x ^double y ^double z ^double w]
            (double (+ x y z w)))]
  (facts "Functor implementation for double vector"
         (instance? clojure.lang.IFn$DD pf1) => true

         (fmap! (fx) pf1) => (->DoubleBlockVector
                              (seq-to-buffer [2 3 4 5]) 4 1)
         (fmap! x pf1) => x

         (fmap! (fx) pf2 (fy))
         => (->DoubleBlockVector (seq-to-buffer [3 5 7 9]) 4 1)
         (fmap! x pf2 (fy))
         => x

         (fmap! (fx) pf3 (fy) (fy))
         => (->DoubleBlockVector (seq-to-buffer [5 8 11 14]) 4 1)
         (fmap! x pf3 (fy) (fy)) => x

         (fmap! (fx) pf4 (fy) (fy) (fy))
         => (->DoubleBlockVector (seq-to-buffer [7 11 15 19]) 4 1)
         (fmap! x pf4 (fy) (fy) (fy))
         => x

         (fmap! (fx) + (fy) (fy) (fy) [(fy)])
         => (throws UnsupportedOperationException)))

(let [fx (fn [] (->DoubleBlockVector (seq-to-buffer [1 2 3 4]) 4 1))
      fy (fn [] (->DoubleBlockVector (seq-to-buffer [2 3 4 5 6]) 5 1))
      x (fx)
      pf1 (fn ^double [^double res ^double x]
            (+ x res))
      pf1o (fn [res ^double x]
            (conj res x))
      pf2 (fn ^double [^double res ^double x ^double y]
            (+ res x y))
      pf2o (fn [res ^double x ^double y]
             (conj res [x y]))
      pf3 (fn ^double [^double res ^double x ^double y ^double z]
            (+ res x y z))
      pf3o (fn [res ^double x ^double y ^double z]
             (conj res [x y z]))]
  (facts "Reducible implementation for double vector"

         (freduce x 1.0 pf1) => 11.0
         (freduce x [] pf1o) => [1.0 2.0 3.0 4.0]

         (freduce x 1.0 pf2 (fy)) => 25.0
         (freduce x [] pf2o (fy))
         => [[1.0 2.0] [2.0 3.0] [3.0 4.0] [4.0 5.0]]

         (freduce x 1.0 pf3 (fy) (fy)) => 39.0
         (freduce x [] pf3o (fy) (fy))
         => [[1.0 2.0 2.0] [2.0 3.0 3.0] [3.0 4.0 4.0] [4.0 5.0 5.0]]

         (freduce x 1.0 + (fy) (fy) [(fy)])
         => (throws UnsupportedOperationException)))

(facts "Vector methods."
       (let [x (->DoubleBlockVector (seq-to-buffer [1 5 3]) 3 1)]
         (.dim x) => 3
         (.iamax x) => 1))

(facts "Vector as a sequence"
       (seq (->DoubleBlockVector (seq-to-buffer [1 2 3]) 3 1))
       => '(1.0 2.0 3.0)
       (seq (->DoubleBlockVector (seq-to-buffer [1 0 2 0 3 0]) 3 2))
       => '(1.0 2.0 3.0))

(facts "RealVector methods."
       (let [x (->DoubleBlockVector (seq-to-buffer [1 2 3]) 3 1)
             y (->DoubleBlockVector (seq-to-buffer [2 3 4]) 3 1)
             xc (->DoubleBlockVector (seq-to-buffer [0 0 0]) 3 1)
             yc (->DoubleBlockVector (seq-to-buffer [0 0 0]) 3 1)]
         (.entry x 1) => 2.0
         (.dot x y) => 20.0
         (.nrm2 x) => (Math/sqrt 14.0)
         (.asum x) => 6.0
         (.iamax x) => 2

         (.scal (.copy x xc) 3)
         => (->DoubleBlockVector (seq-to-buffer [3 6 9]) 3 1)

         (.copy x xc) => xc
         x => xc

         (.swap (.copy x xc) (.copy y yc)) => y
         yc => x

         (.axpy (.copy x xc) 4 (.copy y yc))
         => (->DoubleBlockVector (seq-to-buffer [6 11 16]) 3 1)))

(facts "Matrix methods."
       (let [a (->DoubleGeneralMatrix (seq-to-buffer [1 2 3 4 5 6])
                                      2 3 2 COLUMN_MAJOR)]
         (.mrows a) => 2
         (.ncols a) => 3

         (.row a 1) => (->DoubleBlockVector
                        (seq-to-buffer [2 0 4 0 6 0]) 3 2)

         (.col a 1) => (->DoubleBlockVector
                        (seq-to-buffer [3 4]) 2 1)))

(facts "IFn implementation for double general matrix"
       (let [x (->DoubleGeneralMatrix (seq-to-buffer [1 2 3 4 5 6])
                                      2 3 2 COLUMN_MAJOR)]
         (x 1 2) => 6.0
         (x 2 1) => (throws IndexOutOfBoundsException)
         (x -1 3) => (throws IndexOutOfBoundsException)
         (instance? clojure.lang.IFn x) => true
         (.invokePrim x 0 0) => 1.0))

#_(facts "Functor implementation for double general matrix"
         (let [x (->DoubleGeneralMatrix (seq-to-buffer [1 2 3 4 5 6])
                                        2 3 2 COLUMN_MAJOR)]
         (fmap! x inc) => (->DoubleGeneralMatrix
                           (seq-to-buffer [2 3 4 5 6 7])
                           2 3 2 COLUMN_MAJOR)
         (fmap! x inc) => x))

(let [fx (fn [] (->DoubleGeneralMatrix (seq-to-buffer [1 2 3 4 5 6])
                                       2 3 2 COLUMN_MAJOR))
      fy (fn [] (->DoubleGeneralMatrix (seq-to-buffer [2 3 4 5 6 7])
                                       2 3 2 COLUMN_MAJOR))
      x (fx)
      pf1 (fn ^double [^double x]
            (double (+ x 1.0)))
      pf2 (fn ^double [^double x ^double y]
            (double (+ x y)))
      pf3 (fn ^double [^double x ^double y ^double z]
            (double (+ x y z)))
      pf4 (fn ^double [^double x ^double y ^double z ^double w]
            (double (+ x y z w)))]
  (facts "Functor implementation for double general matrix"
         (instance? clojure.lang.IFn$DD pf1) => true

         (fmap! (fx) pf1) => (fy)
         (fmap! x pf1) => x

         (fmap! (fx) pf2 (fy))
         => (->DoubleGeneralMatrix (seq-to-buffer [3 5 7 9 11 13])
                                   2 3 2 COLUMN_MAJOR)
         (fmap! x pf2 (fy))
         => x

         (fmap! (fx) pf3 (fy) (fy))
         => (->DoubleGeneralMatrix (seq-to-buffer [5 8 11 14 17 20])
                                   2 3 2 COLUMN_MAJOR)
         (fmap! x pf3 (fy) (fy)) => x

         (fmap! (fx) pf4 (fy) (fy) (fy))
         => (->DoubleGeneralMatrix (seq-to-buffer [7 11 15 19 23 27])
                                   2 3 2 COLUMN_MAJOR)
         (fmap! x pf4 (fy) (fy) (fy))
         => x

         (fmap! (fx) + (fy) (fy) (fy) [(fy)])
         => (throws UnsupportedOperationException)))

(facts "RealMatrix methods."
       (let [x (->DoubleBlockVector (seq-to-buffer [1 2 3]) 3 1)
             y (->DoubleBlockVector (seq-to-buffer [2 3]) 2 1)
             xc (->DoubleBlockVector (seq-to-buffer [0 0 0]) 3 1)
             yc (->DoubleBlockVector (seq-to-buffer [0 0]) 2 1)
             a (->DoubleGeneralMatrix (seq-to-buffer [1 2 3 4 5 6])
                                      2 3 2 COLUMN_MAJOR)
             b (->DoubleGeneralMatrix (seq-to-buffer [0.1 0.2 0.3 0.4 0.5 0.6])
                                      3 2 3 COLUMN_MAJOR)
             c (->DoubleGeneralMatrix (seq-to-buffer [0 0 0 0])
                                       2 2 2 COLUMN_MAJOR)]

         (.entry a 1 2) => 6.0

         (.mv a 2.0 (.copy x xc) 3.0 (.copy y yc) NO_TRANS)
         => (->DoubleBlockVector (seq-to-buffer [50 65]) 2 1)

         ;;(.mm a 2 b 3 c NO_TRANS NO_TRANS)
         ;;=> (->DoubleGeneralMatrix (seq-to-buffer [0 0 0 0])
           ;;                        2 2 2 COLUMN_MAJOR)
         ;;(.rank a 2.0 y x)
         ;;=> (->DoubleGeneralMatrix (seq-to-buffer [5 8 11 16 17 24]) 2 3 3 COLUMN_MAJOR)

         ))

(set! *warn-on-reflection* true)
