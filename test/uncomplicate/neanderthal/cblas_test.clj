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

(facts "IFn implementation"
       (let [x (->DoubleBlockVector (seq-to-buffer [1 2 3 4]) 4 1)]
         (x 2) => 3.0
         (x 5) => (throws IndexOutOfBoundsException)
         (x -1) => (throws IndexOutOfBoundsException)
         (instance? clojure.lang.IFn x) => true
         (.invokePrim x 0) => 1.0))

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
