(ns uncomplicate.neanderthal.double-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer :all :exclude [entry entry!]]
             [real :refer :all]
             [native :refer :all]
             [math :refer :all]
             [block :refer [double-accessor]]]))

(defn to-buffer [s]
  (.toBuffer double-accessor s))

(facts "Create a double vector."
       (vect? (dv [1 2 3])) => true
       (dv 1 2 3) => (dv [1 2 3])
       (dv 3) => (dv [0 0 0])
       (dv (to-buffer [1 2 3])) => (dv [1 2 3])
       (dv nil) => (throws IllegalArgumentException)
       (dv []) => (dv (to-buffer []))
       (dv 3) => (zero (dv [1 2 3]))
;;         (dv (long-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ;;(dv (object-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         )

(facts "General vector functions."
       (dim (dv [1 2 3])) => 3
       (dim (dv [])) => 0

       (entry (dv [1 2 3]) 1) => 2.0
       (entry (dv []) 0) => (throws IndexOutOfBoundsException)

       (subvector (dv 1 2 3 4 5 6) 1 3) => (dv 2 3 4)
       (subvector (dv 1 2 3) 2 3)
       => (throws IllegalArgumentException))

;;================ BLAS functions =========================================
(facts "BLAS 1 dot."
       (dot (dv [1 2 3]) (dv [10 30 -10])) => 40.0
       (dot (dv [1 2 3]) nil) => (throws IllegalArgumentException)
       (dot (dv []) (dv [])) => 0.0
       (dot (dv [1]) (dge 3 3)) => (throws IllegalArgumentException)
       (dot (dv [1 2]) (dv [1 2 3]))
       => (throws IllegalArgumentException))

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

(facts "BLAS 1 rot!"
       (let [x (dv [1 2 3 4 5])
             y (dv [-1 -2 -3 -4 -5])]
         (do (rot! x y 1 0) [x y]) => [(dv 1 2 3 4 5) (dv -1 -2 -3 -4 -5)]
         (do (rot! x y 0.5 (/ (sqrt 3) 2.0)) [x y])
         => [(dv -0.3660254037844386 -0.7320508075688772 -1.098076211353316
                 -1.4641016151377544 -1.8301270189221928)
             (dv -1.3660254037844386 -2.732050807568877 -4.098076211353316
                 -5.464101615137754 -6.830127018922193)]
         (rot! x y (/ (sqrt 3) 2.0)  0.5) => x
         (rot! x (dv 1) 0.5 0.5) => (throws IllegalArgumentException)
         (rot! x y 300 0.3) => (throws IllegalArgumentException)
         (rot! x y 0.5 0.5) => (throws IllegalArgumentException)))

(facts "BLAS 1 rotg!"
       (let [x (dv 1 2 3 4)]
         (rotg! x) => x)
       (rotg! (dv 0 0 0 0)) => (dv 0 0 1 0)
       (rotg! (dv 0 2 0 0)) => (dv 2 1 0 1)
       (rotg! (dv 6 -8 0 0))
       => (dv -9.999999999999998 -1.6666666666666665
              -0.6000000000000001 0.8000000000000002))

(facts "BLAS 1 rotm! and rotmg!"
       "TODO: Leave this for later since I am lacking good examples right now.")

(facts "BLAS 1 swap!"
       (let [x (dv [1 2 3])]
         (swp! x (dv [10 20 30])) => (dv [10 20 30])
         (swp! x nil) => (throws IllegalArgumentException)
         (identical? (swp! x (dv [10 20 30])) x) => true
         (swp! x (dv [10 20]))
         => (throws IllegalArgumentException))

       (let [y (dv [0 0 0])]
           (swp! (dv [1 2 3]) y)
           y)
       => (dv [1 2 3]))

(facts "BLAS 1 scal!"
       (let [x (dv [1 2 3])]()
         (identical? (scal! 3 x) x) => true)
       (scal! 3 (dv [1 -2 3])) => (dv [3 -6 9])
       (scal! 3 (dv [])) => (dv []))

(facts "BLAS 1 copy!"
       (copy! (dv [10 20 30]) (dv [1 2 3])) => (dv [10 20 30])
       (copy! (dv [1 2 3]) nil) => (throws IllegalArgumentException)
       (copy! (dv [10 20 30]) (dv [1]))
       => (throws IllegalArgumentException))

(facts "BLAS 1 copy"
       (copy (dv 1 2)) => (dv 1 2)
       (let [x (dv 1 2)]
         (identical? (copy x) x) => false))

(facts "BLAS 1 axpy!"
       (let [y (dv [1 2 3])]
         (axpy! 2.0 (dv [10 20 30]) y) => (dv [21 42 63])
         (identical? (axpy! 3.0 (dv [1 2 3]) y) y) => true
         (axpy! 2.0 (dv [1 2]) y) => (throws IllegalArgumentException)
         (axpy! (dv [10 20 30]) (dv [1 2 3])) => (dv [11 22 33]))

       (axpy! 2 (dv 1 2 3) (dv 1 2 3) (dv 2 3 4) 4.0 (dv 5 6 7))
       => (dv 25 33 41)
       (axpy! 2 (dv 1 2 3) (dv 1 2 3) (dv 2 3 4) 4.0)
       => (throws IllegalArgumentException)
       (axpy! (dv 1 2 3) (dv 1 2 3) (dv 1 2 3) (dv 1 2 3) (dv 1 2 3))
       => (axpy! 5 (dv 1 2 3) (dv 3))
       (axpy! 4 "af" (dv 1 2 3) 3 "b" "c")
       => (throws IllegalArgumentException)
       (let [y (dv [1 2 3])]
         (axpy! 2 (dv 1 2 3) y (dv 2 3 4) 4.0 (dv 5 6 7)) => y))

(facts "BLAS1 axpy"
       (let [y (dv [1 2 3])]
         (axpy 2.0 (dv [10 20 30])) => (throws IllegalArgumentException)
         (identical? (axpy 3.0 (dv [1 2 3]) y) y) => false
         (axpy (dv 1 2 3) (dv 2 3 4) (dv 3 4 5) (dv 3)) => (dv 6 9 12)
         (axpy 2 (dv 1 2 3) (dv 2 3 4)) => (dv 4 7 10)))

;; ================= Real Matrix functions ===========================
(facts "Create a double matrix."
       (let [a (dge 2 3 [1 2 3 4 5 6])]
         (dge 2 3 [1 2 3 4 5 6]) => a
         (dge 2 3 (to-buffer [1 2 3 4 5 6])) => a
         (dge 2 3 nil) => (throws IllegalArgumentException)
         (dge 0 0 []) => (dge 0 0 (to-buffer []))
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
       (entry (dge 0 0) 0 0) => (throws IndexOutOfBoundsException)
       (entry (dge 2 3 [1 2 3 4 5 6]) -1 2)
       => (throws IndexOutOfBoundsException)
       (entry (dge 2 3 [1 2 3 4 5 6]) 2 1)
       => (throws IndexOutOfBoundsException)

       (row (dge 2 3 [1 2 3 4 5 6]) 1) => (dv 2 4 6)
       (row (dge 2 3 [1 2 3 4 5 6]) 4)
       => (throws IndexOutOfBoundsException)

       (col (dge 2 3 [1 2 3 4 5 6]) 1) => (dv 3 4)
       (col (dge 2 3 [1 2 3 4 5 6]) 4)
       => (throws IndexOutOfBoundsException)

       (submatrix (dge 3 4 (range 12)) 1 2 2 1)
       => (dge 2 1 [7 8])
       (submatrix (dge 3 4 (range 12)) 2 3)
       => (dge 2 3 [0 1 3 4 6 7])
       (submatrix (dge 3 4 (range 12)) 3 4)
       => (dge 3 4 (range 12))
       (submatrix (dge 3 4 (range 12)) 1 1 3 3)
       => (throws IndexOutOfBoundsException))

;; ====================== BLAS 2 ===============================
(facts "BLAS 2 mv!"
       (mv! 2.0 (dge 3 2 [1 2 3 4 5 6]) (dv 1 2 3) 3 (dv [1 2 3 4]))
       => (throws IllegalArgumentException)

       (let [y (dv [1 2 3])]
         (mv! 2 (dge 3 2 [1 2 3 4 5 6]) (dv 1 2) 3 y)
         => y)

       (mv! (dge 2 3 [1 10 2 20 3 30]) (dv 7 0 4) (dv 0 0))
       => (dv 19 190)

       (mv! 2.0 (dge 2 3 [1 2 3 4 5 6]) (dv 1 2 3) 3.0 (dv 1 2))
       => (dv 47 62)

       (mv! 2.0 (dge 2 3 [1 2 3 4 5 6]) (dv 1 2 3) 0.0 (dv 0 0))
       => (mv 2.0 (dge 2 3 [1 2 3 4 5 6]) (dv 1 2 3))

       (mv! (trans (dge 3 2 [1 3 5 2 4 6])) (dv 1 2 3) (dv 0 0))
       => (mv! (dge 2 3 [1 2 3 4 5 6]) (dv 1 2 3) (dv 0 0)))

(facts "BLAS 2 rank!"
       (let [a (dge 2 3 [1 2 3 4 5 6])]
         (rank! 2.0 (dv 1 2) (dv 1 2 3) a) => a)

       (rank! (dv 3 2 1 4) (dv 1 2 3) (dge 4 3 [1 2 3 4 2 2 2 2 3 4 2 1]))
       => (dge 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

       (rank! (dv 1 2) (dv 1 2 3) (dge 2 2 [1 2 3 5]))
       => (throws IllegalArgumentException)

       ;;(rank (dv 1 2) (dv 1 2 3)) => (dge 2 3 [1 2 2 4 3 6])
       )

;; ====================== BLAS 3 ================================
(facts "BLAS 3 mm!"
       (mm! 2.0 (dge 3 2 [1 2 3 4 5 6]) (dge 3 2 [1 2 3 4 5 6])
            3.0 (dge 3 2 [1 2 3 4 5 6]))
       => (throws IllegalArgumentException)

       (mm! 2.0 (dge 3 2 [1 2 3 4 5 6]) (dge 2 3 [1 2 3 4 5 6])
            3.0 (dge 3 2 [1 2 3 4 5 6]))
       => (throws IllegalArgumentException)

      (let [c (dge 2 2 [1 2 3 4])]
        (mm! 2.0 (dge 2 3 [1 2 3 4 5 6]) (dge 3 2 [1 2 3 4 5 6]) 3.0 c)
        => c)

      (mm! 2.0 (dge 2 3 [1 2 3 4 5 6]) (dge 3 2 [1 2 3 4 5 6])
           3.0 (dge 2 2 [1 2 3 4]))
       => (dge 2 2 [47 62 107 140])

       (mm! 2.0 (trans (dge 3 2 [1 3 5 2 4 6])) (trans (dge 2 3 [1 4 2 5 3 6]))
            3.0  (dge 2 2 [1 2 3 4]))
       => (dge 2 2 [47 62 107 140]))

(facts "mm"
       (mm (dge 3 2 (range 6)) (dge 2 3 (range 6)))
       => (dge 3 3 [3 4 5 9 14 19 15 24 33]))
