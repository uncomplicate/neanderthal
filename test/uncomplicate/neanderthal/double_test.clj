(ns uncomplicate.neanderthal.double-test
  (:require [midje.sweet :refer [facts throws =>]]
            [vertigo
             [core :refer [wrap]]
             [structs :refer [float64]]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [real :refer :all]
             [protocols :as p]])
  (:import [uncomplicate.neanderthal.cblas 
            DoubleBlockVector DoubleGeneralMatrix]))

(facts "Create a double vector."
       (let [x (DoubleBlockVector. (seq-to-buffer [1 2 3]) 3 1)]
         (dv [1 2 3]) => x
         (dv 1 2 3) => (dv [1 2 3])
         (dv 3) => (dv [0 0 0])
         (dv (seq-to-buffer [1 2 3])) => x
         (dv nil) => (throws IllegalArgumentException)
         (dv []) => (dv (seq-to-buffer []))
         (dv 3) => (zero x)
;;         (dv (long-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ;;(dv (object-array [1 3])) 
         ;;=> (throws IllegalArgumentException)
))

(facts "General vector entryents."
       (dim (dv [1 2 3])) => 3
       (dim (dv [])) => 0
       
       (entry (dv [1 2 3]) 1) => 2.0
       (entry (dv []) 0) => (throws IndexOutOfBoundsException))

(facts "BLAS 1 dot."
       (dot (dv [1 2 3]) (dv [10 30 -10])) => 40.0
       (dot (dv [1 2 3]) nil) => (throws NullPointerException)
       (dot (dv []) (dv [])) => 0.0
       (dot (dv [1]) (dge 3 3)) => (throws ClassCastException)
       (dot (dv [1 2]) (dv [1 2 3])) => (throws IllegalArgumentException))

(facts "BLAS 1 nrm2."
       (nrm2 (dv [1 2 3]))  => (Math/sqrt 14)
       (nrm2 (dv [])) => 0.0)

(facts "BLAS 1 asum."
       (asum (dv [1 2 -3]))  => 6.0
       (asum (dv [])) => 0.0)

(facts "BLAS 1 iamax"
       (iamax (dv [1 2 7 7 6 2 -10 10])) => 6
       (iamax (dv [])) => 0
       (iamax (dv [4 6 7 3])) => 2)

(facts "BLAS 1 swapv!"
       (let [x (dv [1 2 3])]
         (swapv! x (dv [10 20 30])) => (dv [10 20 30])
         (swapv! x nil) => (throws NullPointerException)
         (identical? (swapv! x (dv [10 20 30])) x) => true
         (swapv! x (dv [10 20])) 
         => (throws IllegalArgumentException))
       
       (let [y (dv [0 0 0])]
           (swapv! (dv [1 2 3]) y)
           y) 
       => (dv [1 2 3]))

(facts "BLAS 1 scal!"
       (let [x (dv [1 2 3])]()
         (identical? (scal! 3 x) x) => true)
       (scal! 3 (dv [1 -2 3])) => (dv [3 -6 9])
       (scal! 3 (dv [])) => (dv []))

(facts "BLAS 1 copy!"
       (copy! (dv [10 20 30]) (dv [1 2 3])) => (dv [10 20 30])
       (copy! (dv [1 2 3]) nil) => (throws NullPointerException)
       (copy! (dv [10 20 30]) (dv [1])) 
       => (throws IllegalArgumentException))

(facts "BLAS 1 copy"
       (copy (dv 1 2)) => (dv 1 2)
       (let [x (dv 1 2)]
         (identical? (copy x) x) => false))

(facts "BLAS 1 axpy!"
       (let [y (dv [1 2 3])]
         (axpy! y 2.0 (dv [10 20 30])) => (dv [21 42 63])
         (identical? (axpy! y 3.0 (dv [1 2 3])) y) => true
         (axpy! y 2.0 (dv [1 2])) => (throws IllegalArgumentException)
         (axpy! (dv [1 2 3]) (dv [10 20 30])) => (dv [11 22 33]))

       (axpy! (dv 1 2 3) 2 (dv 1 2 3) (dv 2 3 4) 4.0 (dv 5 6 7))
       => (dv 25 33 41)
       (axpy! (dv 1 2 3) 2 (dv 1 2 3) (dv 2 3 4) 4.0)
       => (throws NullPointerException)
       (axpy! (dv 1 2 3) (dv 1 2 3) (dv 1 2 3) (dv 1 2 3) (dv 1 2 3))
       => (axpy! (dv 3) 5 (dv 1 2 3))
       (axpy! (dv 1 2 3) 4 "af" 3 "b" "c")
       => (throws ClassCastException)
       (let [y (dv [1 2 3])]
         (axpy! y 2 (dv 1 2 3) (dv 2 3 4) 4.0 (dv 5 6 7)) => y))

(facts "BLAS1 axpy"
       (let [y (dv [1 2 3])]
         (axpy 2.0 (dv [10 20 30])) => (dv 20 40 60)
         (identical? (axpy y 3.0 (dv [1 2 3])) y) => false
         (axpy (dv 1 2 3) (dv 2 3 4) (dv 3 4 5)) => (dv 6 9 12)
         (axpy 2 (dv 1 2 3) (dv 2 3 4)) => (dv 4 7 10)))

(facts "Create a double matrix."
       (let [a (DoubleGeneralMatrix. 
                (seq-to-buffer [1 2 3 4 5 6]) 2 3 2 p/COLUMN_MAJOR)]
         (dge 2 3 [1 2 3 4 5 6]) => a
         (dge 2 3 (seq-to-buffer [1 2 3 4 5 6])) => a
         (dge 2 3 nil) => (throws IllegalArgumentException)
         (dge 0 0 []) => (dge 0 0 (seq-to-buffer []))
         ;;(dge 2 1 (long-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ;;(dge 2 1 (object-array [1 3]))
         ;;=> (throws IllegalArgumentException)
))

(facts "General Matrix methods."
       (ncols (dge 2 3 [1 2 3 4 5 6])) => 3
       (ncols (dge 0 0 [])) => 0

       (mrows (dge 2 3)) => 2
       (mrows (dge 0 0)) => 0

       (entry (dge 2 3 [1 2 3 4 5 6]) 0 1) => 3.0
       (entry (dge 0 0) 0 0) => (throws IndexOutOfBoundsException))

(facts "BLAS 2 mv!"
       (mv! (dv [1 2 3 4]) 2.0 (dge 3 2 [1 2 3 4 5 6]) (dv 1 2 3) 3)
       => (throws IllegalArgumentException)
       
       (let [y (dv [1 2 3])]
         (mv! y 2 (dge 3 2 [1 2 3 4 5 6]) (dv 1 2) 3)
         => y)
       
       (mv! (dv 0 0) (dge 2 3 [1 10 2 20 3 30]) (dv 7 0 4))
       => (dv 19 190)

       (mv! (dv 1 2) 2.0 (dge 2 3 [1 2 3 4 5 6]) (dv 1 2 3) 3.0)
       => (dv 47 62)

       (mv! (dv 0 0) 2.0 (dge 2 3 [1 2 3 4 5 6]) (dv 1 2 3) 0.0)
       (mv 2.0 (dge 2 3 [1 2 3 4 5 6]) (dv 1 2 3)))

(facts "BLAS 3 mm!"
       (mm! (dge 3 2 [1 2 3 4 5 6]) 
            2.0 (dge 3 2 [1 2 3 4 5 6]) (dge 3 2 [1 2 3 4 5 6]) 3.0)
       => (throws IllegalArgumentException)

       (mm! (dge 3 2 [1 2 3 4 5 6])
            2.0 (dge 3 2 [1 2 3 4 5 6]) (dge 2 3 [1 2 3 4 5 6]) 3.0)
       => (throws IllegalArgumentException)

      (let [y (dge 2 2 [1 2 3 4])]
         (mm! y 2.0 (dge 2 3 [1 2 3 4 5 6]) 
              (dge 3 2 [1 2 3 4 5 6]) 3.0)
         => y)

       (mm! (dge 2 2 [1 2 3 4])
            2.0 (dge 2 3 [1 2 3 4 5 6])
            (dge 3 2 [1 2 3 4 5 6]) 3.0)
       => (dge 2 2 [47 62 107 140]))
