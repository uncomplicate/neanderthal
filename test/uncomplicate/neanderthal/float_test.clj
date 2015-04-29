(ns uncomplicate.neanderthal.float-test
  (:require [midje.sweet :refer [facts throws =>]]
            [vertigo
             [core :refer [wrap marshal-seq]]
             [structs :refer [float32 unwrap-byte-seq]]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [real :refer :all]
             [math :refer :all]
             [cblas :as cblas]])
  (:import [uncomplicate.neanderthal.cblas
            FloatBlockVector FloatGeneralMatrix]
           [vertigo.bytes ByteSeq]))


(facts "Create a float vector."
       (let [x (FloatBlockVector. (to-buffer Float/BYTES [1 2 3]) 3 1)]
         (sv [1 2 3]) => x
         (sv 1 2 3) => (sv [1 2 3])
         (sv 3) => (sv [0 0 0])
         (sv (to-buffer Float/BYTES [1 2 3]))
         => x
         (sv nil) => (throws IllegalArgumentException)
         (sv []) => (sv (to-buffer Float/BYTES []))
         (sv 3) => (zero x)
;;         (sv (long-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ;;(sv (object-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ))

(facts "General vector functions."
       (dim (sv [1 2 3])) => 3
       (dim (sv [])) => 0

       (entry (sv [1 2 3]) 1) => 2.0
       (entry (sv []) 0) => (throws IndexOutOfBoundsException)

       (subvector (sv 1 2 3 4 5 6) 1 3) => (sv 2 3 4)
       (subvector (sv 1 2 3) 2 3)
       => (throws IllegalArgumentException))

;;================ BLAS functions =========================================
(facts "BLAS 1 dot."
       (dot (sv [1 2 3]) (sv [10 30 -10])) => 40.0
       (dot (sv [1 2 3]) nil) => (throws NullPointerException)
       (dot (sv []) (sv [])) => 0.0
       (dot (sv [1]) (sge 3 3)) => (throws ClassCastException)
       (dot (sv [1 2]) (sv [1 2 3]))
       => (throws IllegalArgumentException))

(facts "BLAS 1 nrm2."
       (nrm2 (sv [1 2 3]))  => (float (Math/sqrt 14))
       (nrm2 (sv [])) => 0.0)

(facts "BLAS 1 asum."
       (asum (sv [1 2 -3]))  => 6.0
       (asum (sv [])) => 0.0)

(facts "BLAS 1 iamax"
       (iamax (sv [1 2 7 7 6 2 -10 10])) => 6
       (iamax (sv [])) => 0
       (iamax (sv [4 6 7 3])) => 2)

(facts "BLAS 1 rot!"
       (let [x (sv [1 2 3 4 5])
             y (sv [-1 -2 -3 -4 -5])]
         (do (rot! x y 1 0) [x y]) => [(sv 1 2 3 4 5) (sv -1 -2 -3 -4 -5)]
         (do (rot! x y 0.5 (/ (sqrt 3) 2.0)) [x y])
         => [(sv -0.3660253882408142 -0.7320507764816284 -1.0980761051177979 -1.4641015529632568 -1.8301267623901367)
             (sv -1.366025447845459 -2.732050895690918 -4.098075866699219 -5.464101791381836 -6.830126762390137)]
         (rot! x y (/ (sqrt 3) 2.0)  0.5) => x
         (rot! x (sv 1) 0.5 0.5) => (throws IllegalArgumentException)
         (rot! x y 300 0.3) => (throws IllegalArgumentException)
         (rot! x y 0.5 0.5) => (throws IllegalArgumentException)))

(facts "BLAS 1 rotg!"
       (let [x (sv 1 2 3 4)]
         (rotg! x) => x)
       (rotg! (sv 0 0 0 0)) => (sv 0 0 1 0)
       (rotg! (sv 0 2 0 0)) => (sv 2 1 0 1)
       (rotg! (sv 6 -8 0 0))
       => (sv -10.000000953674316 -1.6666667461395264 -0.5999999642372131 0.7999999523162842))

(facts "BLAS 1 rotm! and rotmg!"
       "TODO: Leave this for later since I am lacking good examples right now.")

(facts "BLAS 1 swap!"
       (let [x (sv [1 2 3])]
         (swp! x (sv [10 20 30])) => (sv [10 20 30])
         (swp! x nil) => (throws NullPointerException)
         (identical? (swp! x (sv [10 20 30])) x) => true
         (swp! x (sv [10 20]))
         => (throws IllegalArgumentException))

       (let [y (sv [0 0 0])]
           (swp! (sv [1 2 3]) y)
           y)
       => (sv [1 2 3]))

(facts "BLAS 1 scal!"
       (let [x (sv [1 2 3])]()
         (identical? (scal! 3 x) x) => true)
       (scal! 3 (sv [1 -2 3])) => (sv [3 -6 9])
       (scal! 3 (sv [])) => (sv []))

(facts "BLAS 1 copy!"
       (copy! (sv [10 20 30]) (sv [1 2 3])) => (sv [10 20 30])
       (copy! (sv [1 2 3]) nil) => (throws NullPointerException)
       (copy! (sv [10 20 30]) (sv [1]))
       => (throws IllegalArgumentException))

(facts "BLAS 1 copy"
       (copy (sv 1 2)) => (sv 1 2)
       (let [x (sv 1 2)]
         (identical? (copy x) x) => false))

(facts "BLAS 1 axpy!"
       (let [y (sv [1 2 3])]
         (axpy! y 2.0 (sv [10 20 30])) => (sv [21 42 63])
         (identical? (axpy! y 3.0 (sv [1 2 3])) y) => true
         (axpy! y 2.0 (sv [1 2])) => (throws IllegalArgumentException)
         (axpy! (sv [1 2 3]) (sv [10 20 30])) => (sv [11 22 33]))

       (axpy! (sv 1 2 3) 2 (sv 1 2 3) (sv 2 3 4) 4.0 (sv 5 6 7))
       => (sv 25 33 41)
       (axpy! (sv 1 2 3) 2 (sv 1 2 3) (sv 2 3 4) 4.0)
       => (throws NullPointerException)
       (axpy! (sv 1 2 3) (sv 1 2 3) (sv 1 2 3) (sv 1 2 3) (sv 1 2 3))
       => (axpy! (sv 3) 5 (sv 1 2 3))
       (axpy! (sv 1 2 3) 4 "af" 3 "b" "c")
       => (throws ClassCastException)
       (let [y (sv [1 2 3])]
         (axpy! y 2 (sv 1 2 3) (sv 2 3 4) 4.0 (sv 5 6 7)) => y))

(facts "BLAS1 axpy"
       (let [y (sv [1 2 3])]
         (axpy 2.0 (sv [10 20 30])) => (sv 20 40 60)
         (identical? (axpy y 3.0 (sv [1 2 3])) y) => false
         (axpy (sv 1 2 3) (sv 2 3 4) (sv 3 4 5)) => (sv 6 9 12)
         (axpy 2 (sv 1 2 3) (sv 2 3 4)) => (sv 4 7 10)))

;; ================= Real Matrix functions ===========================
(facts "Create a float matrix."
       (let [a (FloatGeneralMatrix.
                (to-buffer Float/BYTES [1 2 3 4 5 6]) 2 3 2 cblas/DEFAULT_ORDER)]
         (sge 2 3 [1 2 3 4 5 6]) => a
         (sge 2 3 (to-buffer Float/BYTES [1 2 3 4 5 6])) => a
         (sge 2 3 nil) => (throws IllegalArgumentException)
         (sge 0 0 []) => (sge 0 0 (to-buffer Float/BYTES []))
         ;;(sge 2 1 (long-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ;;(sge 2 1 (object-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ))

(facts "General Matrix methods."
       (ncols (sge 2 3 [1 2 3 4 5 6])) => 3
       (ncols (sge 0 0 [])) => 0

       (mrows (sge 2 3)) => 2
       (mrows (sge 0 0)) => 0

       (entry (sge 2 3 [1 2 3 4 5 6]) 0 1) => 3.0
       (entry (sge 0 0) 0 0) => (throws IndexOutOfBoundsException)
       (entry (sge 2 3 [1 2 3 4 5 6]) -1 2)
       => (throws IndexOutOfBoundsException)
       (entry (sge 2 3 [1 2 3 4 5 6]) 2 1)
       => (throws IndexOutOfBoundsException)

       (row (sge 2 3 [1 2 3 4 5 6]) 1) => (sv 2 4 6)
       (row (sge 2 3 [1 2 3 4 5 6]) 4)
       => (throws IndexOutOfBoundsException)

       (col (sge 2 3 [1 2 3 4 5 6]) 1) => (sv 3 4)
       (col (sge 2 3 [1 2 3 4 5 6]) 4)
       => (throws IndexOutOfBoundsException)

       (submatrix (sge 3 4 (range 12)) 1 2 2 1)
       => (sge 2 1 [7 8])
       (submatrix (sge 3 4 (range 12)) 2 3)
       => (sge 2 3 [0 1 3 4 6 7])
       (submatrix (sge 3 4 (range 12)) 3 4)
       => (sge 3 4 (range 12))
       (submatrix (sge 3 4 (range 12)) 1 1 3 3)
       => (throws IndexOutOfBoundsException))

;; ====================== BLAS 2 ===============================
(facts "BLAS 2 mv!"
       (mv! (sv [1 2 3 4]) 2.0 (sge 3 2 [1 2 3 4 5 6]) (sv 1 2 3) 3)
       => (throws IllegalArgumentException)

       (let [y (sv [1 2 3])]
         (mv! y 2 (sge 3 2 [1 2 3 4 5 6]) (sv 1 2) 3)
         => y)

       (mv! (sv 0 0) (sge 2 3 [1 10 2 20 3 30]) (sv 7 0 4))
       => (sv 19 190)

       (mv! (sv 1 2) 2.0 (sge 2 3 [1 2 3 4 5 6]) (sv 1 2 3) 3.0)
       => (sv 47 62)

       (mv! (sv 0 0) 2.0 (sge 2 3 [1 2 3 4 5 6]) (sv 1 2 3) 0.0)
       => (mv 2.0 (sge 2 3 [1 2 3 4 5 6]) (sv 1 2 3))

       (mv! (sv 0 0) (trans (sge 3 2 [1 3 5 2 4 6])) (sv 1 2 3))
       => (mv! (sv 0 0) (sge 2 3 [1 2 3 4 5 6]) (sv 1 2 3)))

(facts "BLAS 2 rank!"
       (let [a (sge 2 3 [1 2 3 4 5 6])]
         (rank! a 2.0 (sv 1 2) (sv 1 2 3)) => a)

       (rank! (sge 4 3 [1 2 3 4 2 2 2 2 3 4 2 1])
              (sv 3 2 1 4) (sv 1 2 3))
       => (sge 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

       (rank! (sge 2 2 [1 2 3 5])
             (sv 1 2) (sv 1 2 3))
       => (throws IllegalArgumentException)

       (rank (sv 1 2) (sv 1 2 3)) => (sge 2 3 [1 2 2 4 3 6]))

;; ====================== BLAS 3 ================================
(facts "BLAS 3 mm!"
       (mm! (sge 3 2 [1 2 3 4 5 6])
            2.0 (sge 3 2 [1 2 3 4 5 6]) (sge 3 2 [1 2 3 4 5 6]) 3.0)
       => (throws IllegalArgumentException)

       (mm! (sge 3 2 [1 2 3 4 5 6])
            2.0 (sge 3 2 [1 2 3 4 5 6]) (sge 2 3 [1 2 3 4 5 6]) 3.0)
       => (throws IllegalArgumentException)

      (let [c (sge 2 2 [1 2 3 4])]
         (mm! c 2.0 (sge 2 3 [1 2 3 4 5 6])
              (sge 3 2 [1 2 3 4 5 6]) 3.0)
         => c)

       (mm! (sge 2 2 [1 2 3 4])
            2.0 (sge 2 3 [1 2 3 4 5 6])
            (sge 3 2 [1 2 3 4 5 6]) 3.0)
       => (sge 2 2 [47 62 107 140])

       (mm! (sge 2 2 [1 2 3 4])
            2.0 (trans (sge 3 2 [1 3 5 2 4 6]))
            (trans (sge 2 3 [1 4 2 5 3 6])) 3.0)
       => (sge 2 2 [47 62 107 140]))

(facts "mm"
       (mm (sge 3 2 (range 6)) (sge 2 3 (range 6)))
       => (sge 3 3 [3 4 5 9 14 19 15 24 33]))
