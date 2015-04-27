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
            FloatBlockVector]
           [vertigo.bytes ByteSeq]))


(facts "Create a float vector."
       (let [x (FloatBlockVector. (to-buffer float32 [1 2 3]) 3 1)]
         (fv [1 2 3]) => x
         (fv 1 2 3) => (fv [1 2 3])
         (fv 3) => (fv [0 0 0])
         (fv (to-buffer float32 [1 2 3]))
         => x
         (fv nil) => (throws IllegalArgumentException)
         (fv []) => (fv (to-buffer float32 []))
         (fv 3) => (zero x)
;;         (fv (long-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ;;(fv (object-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ))

(facts "General vector functions."
       (dim (fv [1 2 3])) => 3
       (dim (fv [])) => 0

       (entry (fv [1 2 3]) 1) => 2.0
       (entry (fv []) 0) => (throws IndexOutOfBoundsException)

       (subvector (fv 1 2 3 4 5 6) 1 3) => (fv 2 3 4)
       (subvector (fv 1 2 3) 2 3)
       => (throws IllegalArgumentException))

;;================ BLAS functions =========================================
(facts "BLAS 1 dot."
       (dot (fv [1 2 3]) (fv [10 30 -10])) => 40.0
       (dot (fv [1 2 3]) nil) => (throws NullPointerException)
       (dot (fv []) (fv [])) => 0.0
       (dot (fv [1]) (dge 3 3)) => (throws ClassCastException)
       (dot (fv [1 2]) (fv [1 2 3]))
       => (throws IllegalArgumentException))

(facts "BLAS 1 nrm2."
       (nrm2 (fv [1 2 3]))  => (float (Math/sqrt 14))
       (nrm2 (fv [])) => 0.0)

(facts "BLAS 1 asum."
       (asum (fv [1 2 -3]))  => 6.0
       (asum (fv [])) => 0.0)

(facts "BLAS 1 iamax"
       (iamax (fv [1 2 7 7 6 2 -10 10])) => 6
       (iamax (fv [])) => 0
       (iamax (fv [4 6 7 3])) => 2)

(facts "BLAS 1 rot!"
       (let [x (fv [1 2 3 4 5])
             y (fv [-1 -2 -3 -4 -5])]
         (do (rot! x y 1 0) [x y]) => [(fv 1 2 3 4 5) (fv -1 -2 -3 -4 -5)]
         (do (rot! x y 0.5 (/ (sqrt 3) 2.0)) [x y])
         => [(fv -0.3660253882408142 -0.7320507764816284 -1.0980761051177979 -1.4641015529632568 -1.8301267623901367)
             (fv -1.366025447845459 -2.732050895690918 -4.098075866699219 -5.464101791381836 -6.830126762390137)]
         (rot! x y (/ (sqrt 3) 2.0)  0.5) => x
         (rot! x (fv 1) 0.5 0.5) => (throws IllegalArgumentException)
         (rot! x y 300 0.3) => (throws IllegalArgumentException)
         (rot! x y 0.5 0.5) => (throws IllegalArgumentException)))

(facts "BLAS 1 rotg!"
       (let [x (fv 1 2 3 4)]
         (rotg! x) => x)
       (rotg! (fv 0 0 0 0)) => (fv 0 0 1 0)
       (rotg! (fv 0 2 0 0)) => (fv 2 1 0 1)
       (rotg! (fv 6 -8 0 0))
       => (fv -10.000000953674316 -1.6666667461395264 -0.5999999642372131 0.7999999523162842))

(facts "BLAS 1 rotm! and rotmg!"
       "TODO: Leave this for later since I am lacking good examples right now.")

(facts "BLAS 1 swap!"
       (let [x (fv [1 2 3])]
         (swp! x (fv [10 20 30])) => (fv [10 20 30])
         (swp! x nil) => (throws NullPointerException)
         (identical? (swp! x (fv [10 20 30])) x) => true
         (swp! x (fv [10 20]))
         => (throws IllegalArgumentException))

       (let [y (fv [0 0 0])]
           (swp! (fv [1 2 3]) y)
           y)
       => (fv [1 2 3]))

(facts "BLAS 1 scal!"
       (let [x (fv [1 2 3])]()
         (identical? (scal! 3 x) x) => true)
       (scal! 3 (fv [1 -2 3])) => (fv [3 -6 9])
       (scal! 3 (fv [])) => (fv []))

(facts "BLAS 1 copy!"
       (copy! (fv [10 20 30]) (fv [1 2 3])) => (fv [10 20 30])
       (copy! (fv [1 2 3]) nil) => (throws NullPointerException)
       (copy! (fv [10 20 30]) (fv [1]))
       => (throws IllegalArgumentException))

(facts "BLAS 1 copy"
       (copy (fv 1 2)) => (fv 1 2)
       (let [x (fv 1 2)]
         (identical? (copy x) x) => false))

(facts "BLAS 1 axpy!"
       (let [y (fv [1 2 3])]
         (axpy! y 2.0 (fv [10 20 30])) => (fv [21 42 63])
         (identical? (axpy! y 3.0 (fv [1 2 3])) y) => true
         (axpy! y 2.0 (fv [1 2])) => (throws IllegalArgumentException)
         (axpy! (fv [1 2 3]) (fv [10 20 30])) => (fv [11 22 33]))

       (axpy! (fv 1 2 3) 2 (fv 1 2 3) (fv 2 3 4) 4.0 (fv 5 6 7))
       => (fv 25 33 41)
       (axpy! (fv 1 2 3) 2 (fv 1 2 3) (fv 2 3 4) 4.0)
       => (throws NullPointerException)
       (axpy! (fv 1 2 3) (fv 1 2 3) (fv 1 2 3) (fv 1 2 3) (fv 1 2 3))
       => (axpy! (fv 3) 5 (fv 1 2 3))
       (axpy! (fv 1 2 3) 4 "af" 3 "b" "c")
       => (throws ClassCastException)
       (let [y (fv [1 2 3])]
         (axpy! y 2 (fv 1 2 3) (fv 2 3 4) 4.0 (fv 5 6 7)) => y))

(facts "BLAS1 axpy"
       (let [y (fv [1 2 3])]
         (axpy 2.0 (fv [10 20 30])) => (fv 20 40 60)
         (identical? (axpy y 3.0 (fv [1 2 3])) y) => false
         (axpy (fv 1 2 3) (fv 2 3 4) (fv 3 4 5)) => (fv 6 9 12)
         (axpy 2 (fv 1 2 3) (fv 2 3 4)) => (fv 4 7 10)))

;; ================= Real Matrix functions ===========================
#_(facts "Create a float matrix."
       (let [a (FloatGeneralMatrix.
                (to-buffer [1 2 3 4 5 6]) 2 3 2 cblas/DEFAULT_ORDER)]
         (dge 2 3 [1 2 3 4 5 6]) => a
         (dge 2 3 (to-buffer [1 2 3 4 5 6])) => a
         (dge 2 3 nil) => (throws IllegalArgumentException)
         (dge 0 0 []) => (dge 0 0 (to-buffer []))
         ;;(dge 2 1 (long-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ;;(dge 2 1 (object-array [1 3]))
         ;;=> (throws IllegalArgumentException)
         ))

#_(facts "General Matrix methods."
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

       (row (dge 2 3 [1 2 3 4 5 6]) 1) => (fv 2 4 6)
       (row (dge 2 3 [1 2 3 4 5 6]) 4)
       => (throws IndexOutOfBoundsException)

       (col (dge 2 3 [1 2 3 4 5 6]) 1) => (fv 3 4)
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
#_(facts "BLAS 2 mv!"
       (mv! (fv [1 2 3 4]) 2.0 (dge 3 2 [1 2 3 4 5 6]) (fv 1 2 3) 3)
       => (throws IllegalArgumentException)

       (let [y (fv [1 2 3])]
         (mv! y 2 (dge 3 2 [1 2 3 4 5 6]) (fv 1 2) 3)
         => y)

       (mv! (fv 0 0) (dge 2 3 [1 10 2 20 3 30]) (fv 7 0 4))
       => (fv 19 190)

       (mv! (fv 1 2) 2.0 (dge 2 3 [1 2 3 4 5 6]) (fv 1 2 3) 3.0)
       => (fv 47 62)

       (mv! (fv 0 0) 2.0 (dge 2 3 [1 2 3 4 5 6]) (fv 1 2 3) 0.0)
       => (mv 2.0 (dge 2 3 [1 2 3 4 5 6]) (fv 1 2 3))

       (mv! (fv 0 0) (trans (dge 3 2 [1 3 5 2 4 6])) (fv 1 2 3))
       => (mv! (fv 0 0) (dge 2 3 [1 2 3 4 5 6]) (fv 1 2 3)))

#_(facts "BLAS 2 rank!"
       (let [a (dge 2 3 [1 2 3 4 5 6])]
         (rank! a 2.0 (fv 1 2) (fv 1 2 3)) => a)

       (rank! (dge 4 3 [1 2 3 4 2 2 2 2 3 4 2 1])
              (fv 3 2 1 4) (fv 1 2 3))
       => (dge 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

       (rank! (dge 2 2 [1 2 3 5])
             (fv 1 2) (fv 1 2 3))
       => (throws IllegalArgumentException)

       (rank (fv 1 2) (fv 1 2 3)) => (dge 2 3 [1 2 2 4 3 6]))

;; ====================== BLAS 3 ================================
#_(facts "BLAS 3 mm!"
       (mm! (dge 3 2 [1 2 3 4 5 6])
            2.0 (dge 3 2 [1 2 3 4 5 6]) (dge 3 2 [1 2 3 4 5 6]) 3.0)
       => (throws IllegalArgumentException)

       (mm! (dge 3 2 [1 2 3 4 5 6])
            2.0 (dge 3 2 [1 2 3 4 5 6]) (dge 2 3 [1 2 3 4 5 6]) 3.0)
       => (throws IllegalArgumentException)

      (let [c (dge 2 2 [1 2 3 4])]
         (mm! c 2.0 (dge 2 3 [1 2 3 4 5 6])
              (dge 3 2 [1 2 3 4 5 6]) 3.0)
         => c)

       (mm! (dge 2 2 [1 2 3 4])
            2.0 (dge 2 3 [1 2 3 4 5 6])
            (dge 3 2 [1 2 3 4 5 6]) 3.0)
       => (dge 2 2 [47 62 107 140])

       (mm! (dge 2 2 [1 2 3 4])
            2.0 (trans (dge 3 2 [1 3 5 2 4 6]))
            (trans (dge 2 3 [1 4 2 5 3 6])) 3.0)
       => (dge 2 2 [47 62 107 140]))

#_(facts "mm"
       (mm (dge 3 2 (range 6)) (dge 2 3 (range 6)))
       => (dge 3 3 [3 4 5 9 14 19 15 24 33]))
