;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.real-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.neanderthal
             [protocols :refer [data-accessor]]
             [core :refer :all]
             [math :refer :all]])
  (:import [uncomplicate.neanderthal.protocols RealBufferAccessor]))

(defn seq-to-buffer [^RealBufferAccessor acc s]
  (let [b (.createDataSource acc (count s))]
    (reduce (fn [^long i ^double e] (do (.set acc b i e) (inc i))) 0 s)
    b))

(defn test-group [factory]
  (facts "Group methods."
         (let [x (vctr factory [1 2 3])
               y (vctr factory [2 3 4])]

           (zero x) => (vctr factory [0 0 0])
           (identical? x (zero x)) => false)

         (let [a (ge factory 2 3 [1 2 3 4 5 6])
               ac (ge factory 2 3 [0 0 0 0 0 0])]
           (zero a) => ac
           (identical? a (zero a)) => false)))

(defn test-ge [factory]
  (facts "General Matrix methods."
         (let [a (ge factory 2 3 [1 2 3 4 5 6])]
           (mrows a) => 2
           (ncols a) => 3

           (row a 1) => (vctr factory [2 4 6])

           (col a 1) => (vctr factory [3 4]))))

(defn test-tr [factory]
  (facts "Triangular Matrix methods."
         (let [a (ge factory 5 3 (range 15))
               a-upper (subtriangle a {:uplo :upper})
               a-lower (subtriangle a)]
           (= 3 (mrows a-upper) (ncols a-lower)) => true
           (row a-upper 0) => (vctr factory [0 5 10])
           (row a-upper 2) => (vctr factory [12])
           (col a-upper 0) => (vctr factory [0])
           (col a-upper 2) => (vctr factory [10 11 12])
           (row a-lower 0) => (vctr factory [0])
           (row a-lower 2) => (vctr factory [2 7 12])
           (col a-lower 0) => (vctr factory [0 1 2])
           (col a-lower 2) => (vctr factory [12]))))

;; ============= Matrices and Vectors ======================================

(defn test-vector-constructor [factory]
  (facts "Create a real vector."
         (vect? (vctr factory [1 2 3])) => true
         (vctr factory 1 2 3) => (vctr factory [1 2 3])
         (vctr factory 3) => (vctr factory [0 0 0])
         ;;(vctr factory (seq-to-buffer (data-accessor factory) [1 2 3]))
         ;;=> (vctr factory [1 2 3])
         (vctr factory nil) => (throws IllegalArgumentException)
         (dim (vctr factory [])) => 0
         (vctr factory 3) => (zero (vctr factory [1 2 3]))))

(defn test-vector [factory]
  (facts "General vector functions."
         (dim (vctr factory [1 2 3])) => 3
         (dim (vctr factory [])) => 0

         (subvector (vctr factory 1 2 3 4 5 6) 1 3)
         => (vctr factory 2 3 4)
         (subvector (vctr factory 1 2 3) 2 3)
         => (throws IndexOutOfBoundsException)))

(defn test-vector-entry [factory]
  (facts "Vectory entry."
         (entry (vctr factory [1 2 3]) 1) => 2.0
         (entry (vctr factory []) 0)
         => (throws IndexOutOfBoundsException)))

(defn test-vector-entry! [factory]
  (facts "Vectory entry!."
         (entry (entry! (vctr factory [1 2 3]) 1 77.0) 1) => 77.0))

(defn test-vector-bulk-entry! [factory]
  (facts "Vectory entry!."
         (sum (entry! (vctr factory [1 2 3]) 77.0)) => 231.0))

;;================ BLAS functions =========================================

(defn test-dot [factory]
  (facts "BLAS 1 dot."
         (dot (vctr factory [1 2 3]) (vctr factory [10 30 -10]))
         => 40.0
         (dot (vctr factory [1 2 3]) nil)
         => (throws IllegalArgumentException)
         (dot (vctr factory []) (vctr factory [])) => 0.0
         (dot (vctr factory [1]) (ge factory 3 3))
         => (throws ClassCastException)
         (dot (vctr factory [1 2]) (vctr factory [1 2 3]))
         => (throws IllegalArgumentException)))

(defn test-nrm2 [factory]
  (facts "BLAS 1 nrm2."
         (nrm2 (vctr factory [1 2 3]))  => (roughly (Math/sqrt 14))
         (nrm2 (vctr factory [])) => 0.0))

(defn test-asum [factory]
  (facts "BLAS 1 asum."
         (asum (vctr factory [1 2 -3]))  => 6.0
         (asum (vctr factory [])) => 0.0))

(defn test-sum [factory]
  (facts "BLAS Plus 1 sum."
         (sum (vctr factory [1 2 -5]))  => -2.0
         (sum (vctr factory [])) => 0.0))

(defn test-iamax [factory]
  (facts "BLAS 1 iamax"
         (iamax (vctr factory [1 2 7 7 6 2 -12 10])) => 6
         (iamax (vctr factory [])) => 0
         (iamax (vctr factory [4 6 7 3])) => 2))

(defn test-imax [factory]
  (facts "BLAS 1 imax"
         (imax (vctr factory [1 2 7 7 6 2 -10 10])) => 7
         (imax (vctr factory [])) => 0
         (imax (vctr factory [4 6 7 3])) => 2))

(defn test-imin [factory]
  (facts "BLAS 1 imin"
         (imin (vctr factory [1 2 7 7 6 2 -12 10])) => 6
         (imin (vctr factory [])) => 0
         (imin (vctr factory [4 6 7 3])) => 3))

(defn test-rot [factory]
  (facts "BLAS 1 rot!"
         (let [x (vctr factory [1 2 3 4 5])
               y (vctr factory [-1 -2 -3 -4 -5])]
           (do (rot! x y 1 0) [x y])
           => [(vctr factory 1 2 3 4 5)
               (vctr factory -1 -2 -3 -4 -5)]

           (do (rot! x y 0.5 (/ (sqrt 3) 2.0)) [x y]
               (+ (double (nrm2 (axpy -1 x (vctr factory
                                                          -0.3660254037844386
                                                          -0.7320508075688772
                                                          -1.098076211353316
                                                          -1.4641016151377544
                                                          -1.8301270189221928))))
                  (double (nrm2 (axpy -1 y (vctr factory
                                                          -1.3660254037844386
                                                          -2.732050807568877
                                                          -4.098076211353316
                                                          -5.464101615137754
                                                          -6.830127018922193))))))
           => (roughly 0 1e-6)

           (rot! x y (/ (sqrt 3) 2.0)  0.5) => x
           (rot! x (vctr factory 1) 0.5 0.5)
           => (throws IllegalArgumentException)
           (rot! x y 300 0.3) => (throws IllegalArgumentException)
           (rot! x y 0.5 0.5) => (throws IllegalArgumentException))))

(defn test-rotg [factory]
  (facts "BLAS 1 rotg!"
         (let [x (vctr factory 1 2 3 4)]
           (rotg! x) => x)
         (rotg! (vctr factory 0 0 0 0))
         => (vctr factory 0 0 1 0)
         (rotg! (vctr factory 0 2 0 0))
         => (vctr factory 2 1 0 1)
         (nrm2 (axpy -1 (rotg! (vctr factory 6 -8 0 0))
                     (vctr factory -10 -5/3 -3/5 4/5)))
         => (roughly 0 1e-6)))

(defn test-rotm [factory]
  (facts "BLAS 1 rotm!"
         "TODO: Leave this for later since I am lacking good examples right now."))

(defn test-rotmg [factory]
  (facts "BLAS 1 rotmg!"
         "TODO: Leave this for later since I am lacking good examples right now."))

(defn test-swap [factory]
  (facts
   "BLAS 1 swap! vectors"
   (let [x (vctr factory [1 2 3])]
     (swp! x (vctr factory 3)) => x
     (swp! x (vctr factory [10 20 30]))
     => (vctr factory [10 20 30])
     (swp! x nil) => (throws IllegalArgumentException)
     (identical? (swp! x (vctr factory [10 20 30])) x) => true
     (swp! x (vctr factory [10 20]))
     => (throws IllegalArgumentException)))

  (facts
   "BLAS 1 swap! matrices"
   (let [a (ge factory 2 3 [1 2 3 4 5 6])]
     (swp! a (ge factory 2 3)) => a
     (swp! a (ge factory 2 3 [10 20 30 40 50 60]))
     => (ge factory 2 3 [10 20 30 40 50 60])
     (swp! a nil) => (throws IllegalArgumentException)
     (identical? (swp! a (ge factory 2 3 [10 20 30 40 50 60])) a)
     => true
     (swp! a (ge factory 1 2 [10 20]))
     => (throws IllegalArgumentException))))

(defn test-scal [factory]
  (facts "BLAS 1 scal!"
         (let [x (vctr factory [1 2 3])]()
              (identical? (scal! 3 x) x) => true)
         (scal! 3 (vctr factory [1 -2 3]))
         => (vctr factory [3 -6 9])
         (scal! 3 (vctr factory []))
         => (vctr factory [])))

(defn test-vector-copy [factory]
  (facts "BLAS 1 copy!"
         (let [y (vctr factory 3)]
           (copy! (vctr factory [1 2 3]) y) => y
           (copy (vctr factory [1 2 3])) => y)

         (copy! (vctr factory [10 20 30])
                (vctr factory [1 2 3]))
         => (vctr factory [10 20 30])
         (copy! (vctr factory [1 2 3]) nil)
         => (throws IllegalArgumentException)
         (copy! (vctr factory [10 20 30]) (vctr factory [1]))
         => (throws IllegalArgumentException)

         (copy (vctr factory 1 2)) => (vctr factory 1 2)
         (let [x (vctr factory 1 2)]
           (identical? (copy x) x) => false)))

(defn test-axpy [factory]
  (facts "BLAS 1 axpy!"
         (let [y (vctr factory [1 2 3])]
           (axpy! 2.0 (vctr factory [10 20 30]) y)
           => (vctr factory [21 42 63])
           (identical? (axpy! 3.0 (vctr factory [1 2 3]) y) y) => true
           (axpy! 2.0 (vctr factory [1 2]) y)
           => (throws IllegalArgumentException)
           (axpy! (vctr factory [10 20 30])
                  (vctr factory [1 2 3]))
           => (vctr factory [11 22 33]))

         (axpy! 2 (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7))
         => (vctr factory 25 33 41)
         (axpy! 2 (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 2 3 4) 4.0)
         => (throws IllegalArgumentException)
         (axpy! (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 1 2 3))
         => (axpy! 5 (vctr factory 1 2 3) (vctr factory 3))
         (axpy! 4 "af" (vctr factory 1 2 3) 3 "b" "c")
         => (throws IllegalArgumentException)
         (let [y (vctr factory [1 2 3])]
           (axpy! 2 (vctr factory 1 2 3) y
                  (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7))
           => y))

  (facts "BLAS1 axpy"
         (let [y (vctr factory [1 2 3])]
           (axpy 2.0 (vctr factory [10 20 30]))
           => (throws IllegalArgumentException)
           (identical? (axpy 3.0 (vctr factory [1 2 3]) y) y) => false
           (axpy (vctr factory 1 2 3) (vctr factory 2 3 4)
                 (vctr factory 3 4 5) (vctr factory 3))
           => (vctr factory 6 9 12)
           (axpy 2 (vctr factory 1 2 3) (vctr factory 2 3 4))
           => (vctr factory 4 7 10))))

;; ================= Real General Matrix functions =============================

(defn test-ge-constructor [factory]
  (facts "Create a general matrix."
         (let [a (ge factory 2 3 [1 2 3 4 5 6])]
           (ge factory 2 3 [1 2 3 4 5 6]) => a
           ;;(ge factory 2 3 (seq-to-buffer (data-accessor factory)
           ;;[1 2 3 4 5 6]))
           ;;=> a
           (ge factory 2 3 nil) => (zero a)
           ;;(ge factory 0 0 [])
           ;;=> (ge factory 0 0 (seq-to-buffer (data-accessor factory) []))
           )))

(defn test-ge [factory]
  (facts "General Matrix methods."
         (ncols (ge factory 2 3 [1 2 3 4 5 6])) => 3
         (ncols (ge factory 0 0 [])) => 0

         (mrows (ge factory 2 3)) => 2
         (mrows (ge factory 0 0)) => 0

         (row (ge factory 2 3 [1 2 3 4 5 6]) 1)
         => (vctr factory 2 4 6)
         (row (ge factory 2 3 [1 2 3 4 5 6]) 4)
         => (throws IndexOutOfBoundsException)

         (col (ge factory 2 3 [1 2 3 4 5 6]) 1)
         => (vctr factory 3 4)
         (col (ge factory 2 3 [1 2 3 4 5 6]) 4)
         => (throws IndexOutOfBoundsException)

         (submatrix (ge factory 3 4 (range 12)) 1 2 2 1)
         => (ge factory 2 1 [7 8])
         (submatrix (ge factory 3 4 (range 12)) 2 3)
         => (ge factory 2 3 [0 1 3 4 6 7])
         (submatrix (ge factory 3 4 (range 12)) 3 4)
         => (ge factory 3 4 (range 12))
         (submatrix (ge factory 3 4 (range 12)) 1 1 3 3)
         => (throws IndexOutOfBoundsException)))

(defn test-ge-entry [factory]
  (facts "Matrix entry."
         (entry (ge factory 2 3 [1 2 3 4 5 6]) 0 1) => 3.0
         (entry (ge factory 0 0) 0 0)
         => (throws IndexOutOfBoundsException)
         (entry (ge factory 2 3 [1 2 3 4 5 6]) -1 2)
         => (throws IndexOutOfBoundsException)
         (entry (ge factory 2 3 [1 2 3 4 5 6]) 2 1)
         => (throws IndexOutOfBoundsException)))

(defn test-ge-entry! [factory]
  (facts "Matrix entry!."
         (entry (entry! (ge factory 2 3 [1 2 3 4 5 6]) 0 1 88.0) 0 1) => 88.0))

(defn test-ge-bulk-entry! [factory]
  (facts "Matrix entry!."
         (sum (row (entry! (ge factory 2 3 [1 2 3 4 5 6]) 88.0) 1)) => 264.0))

(defn test-ge-copy [factory]
  (facts "BLAS 1 copy! general matrix"
         (let [a (ge factory 2 3)]
           (copy! (ge factory 2 3 [1 2 3 4 5 6]) a) => a
           (copy (ge factory 2 3 [1 2 3 4 5 6])) => a)

         (copy! (ge factory 2 3 [10 20 30 40 50 60])
                (ge factory 2 3 [1 2 3 4 5 6]))
         => (ge factory 2 3 [10 20 30 40 50 60])
         (copy! (ge factory 2 3 [1 2 3 4 5 6]) nil)
         => (throws IllegalArgumentException)
         (copy! (ge factory 2 3 [10 20 30 40 50 60])
                (ge factory 2 2))
         => (throws IllegalArgumentException)

         (copy (vctr factory 1 2)) => (vctr factory 1 2)
         (let [x (vctr factory 1 2)]
           (identical? (copy x) x) => false)))

(defn test-ge-scal [factory]
  (facts "BLAS 1 scal! general matrix"
         (let [x (ge factory 2 3 [1 2 3 4 5 6])]()
              (identical? (scal! 3 x) x) => true)
         (scal! 3 (ge factory 2 3 [1 -2 3 9 8 7]))
         => (ge factory 2 3 [3 -6 9 27 24 21])
         (scal! 3 (submatrix (ge factory 3 2 [1 2 3 4 5 6]) 1 1 1 1))
         => (ge factory 1 1 [15])))

;; ====================== BLAS 2 ===============================

(defn test-mv [factory]
  (facts "BLAS 2 mv!"
         (mv! 2.0 (ge factory 3 2 [1 2 3 4 5 6])
              (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4]))
         => (throws IllegalArgumentException)

         (let [y (vctr factory [1 2 3])]
           (mv! 2 (ge factory 3 2 [1 2 3 4 5 6])
                (vctr factory 1 2) 3 y)
           => y)

         (mv! (ge factory 2 3 [1 10 2 20 3 30])
              (vctr factory 7 0 4) (vctr factory 0 0))
         => (vctr factory 19 190)

         (mv! 2.0 (ge factory 2 3 [1 2 3 4 5 6])
              (vctr factory 1 2 3) 3.0 (vctr factory 1 2))
         => (vctr factory 47 62)

         (mv! 2.0 (ge factory 2 3 [1 2 3 4 5 6])
              (vctr factory 1 2 3) 0.0 (vctr factory 0 0))
         => (mv 2.0 (ge factory 2 3 [1 2 3 4 5 6])
                (vctr factory 1 2 3))

         (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
              (vctr factory [1 3 5])
              3.0 (vctr factory [2 3]))
         => (vctr factory 272 293)

         (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
              (vctr factory [1 3 5])
              3.0 (col (ge factory 2 3 (range 6)) 1))
         => (vctr factory 272 293)

         (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
              (row (ge factory 2 3 (range 6)) 1)
              3.0 (col (ge factory 2 3 (range 6)) 1))
         => (vctr factory 272 293)))

(defn test-mv-transpose [factory]
  (facts "BLAS 2 mv!"
         (mv! (trans (ge factory 3 2 [1 3 5 2 4 6]))
              (vctr factory 1 2 3) (vctr factory 0 0))
         => (mv! (ge factory 2 3 [1 2 3 4 5 6])
                 (vctr factory 1 2 3) (vctr factory 0 0))))

(defn test-rank [factory]
  (facts "BLAS 2 rank!"
         (let [a (ge factory 2 3)]
           (rank! 2.0 (vctr factory 1 2)
                  (vctr factory 1 2 3) a)
           => a)
         (rank! (vctr factory 3 2 1 4)
                (vctr factory 1 2 3)
                (ge factory 4 3 [1 2 3 4 2 2 2 2 3 4 2 1]))
         => (ge factory 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

         (rank! (vctr factory 1 2) (vctr factory 1 2 3)
                (ge factory 2 2 [1 2 3 5]))
         => (throws IllegalArgumentException)))

;; ====================== BLAS 3 ================================

(defn test-mm [factory]
  (facts "BLAS 3 mm!"
         (mm! 2.0 (ge factory 3 2 [1 2 3 4 5 6])
              (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6]))
         => (throws IllegalArgumentException)

         (mm! 2.0 (ge factory 3 2 [1 2 3 4 5 6])
              (ge factory 2 3 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6]))
         => (throws IllegalArgumentException)

         (let [c (ge factory 2 2 [1 2 3 4])]
           (mm! 2.0 (ge factory 2 3 [1 2 3 4 5 6])
                (ge factory 3 2 [1 2 3 4 5 6]) 3.0 c)
           => c)

         (mm! 2.0 (ge factory 2 3 [1 2 3 4 5 6])
              (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 2 2 [1 2 3 4]))
         => (ge factory 2 2 [47 62 107 140])

         (mm! 2.0 (ge factory 3 5 (take 15 (repeat 1)))
              (ge factory 5 3 (take 15 (repeat 1)))
              3.0 (ge factory 3 3 (take 9 (repeat 0))))
         => (ge factory 3 3 (take 9 (repeat 10))))

  (facts "mm"
         (mm (ge factory 3 2 (range 6))
             (ge factory 2 3 (range 6)))
         => (ge factory 3 3 [3 4 5 9 14 19 15 24 33])))

(defn test-mm-row-major [factory]
  (facts "BLAS 3 mm! transposed (row major)"
         (mm! 2.0 (trans (ge factory 3 2 [1 3 5 2 4 6]))
              (trans (ge factory 2 3 [1 4 2 5 3 6]))
              3.0  (ge factory 2 2 [1 2 3 4]))
         => (ge factory 2 2 [47 62 107 140])))

;; ====================== TR Matrix ============================

(defn test-tr-constructor [factory]
  (facts "Create a triangular matrix."
         (let [a (tr factory 3 (range 6))]
           (subtriangle (ge factory 3 3 [0 1 2 0 3 4 0 0 5])) => a
           ;;(ge factory 2 3 (seq-to-buffer (data-accessor factory)
           ;;[1 2 3 4 5 6]))
           ;;=> a
           (tr factory 3 nil) => (zero a)
           ;;(ge factory 0 0 [])
           ;;=> (ge factory 0 0 (seq-to-buffer (data-accessor factory) []))
           )))

(defn test-tr-entry [factory]
  (facts "TR Matrix entry."
         (with-release [a-upper (tr factory 3 (range 6) {:uplo :upper})]
           (entry a-upper 0 1) => 1.0
           (entry a-upper 1 0) => 0.0
           (entry a-upper 1 1) => 2.0
           (entry a-upper 2 0) => 0.0
           (entry a-upper 0 2) => 3.0
           (entry a-upper -1 2) => (throws IndexOutOfBoundsException)
           (entry a-upper 3 2) => (throws IndexOutOfBoundsException))))

;; =========================================================================
(defn test-all [factory]
  (do
    (test-group factory)
    (test-vector-constructor factory)
    (test-vector factory)
    (test-vector-entry factory)
    (test-vector-entry! factory)
    (test-vector-bulk-entry! factory)
    (test-dot factory)
    (test-nrm2 factory)
    (test-asum factory)
    (test-sum factory)
    (test-iamax factory)
    (test-rot factory)
    (test-rotg factory)
    (test-rotm factory)
    (test-rotmg factory)
    (test-swap factory)
    (test-scal factory)
    (test-vector-copy factory)
    (test-axpy factory)
    (test-ge-constructor factory)
    (test-ge factory)
    (test-ge factory)
    (test-ge-entry factory)
    (test-ge-entry! factory)
    (test-ge-bulk-entry! factory)
    (test-ge-copy factory)
    (test-ge-scal factory)
    (test-mv factory)
    (test-mv-transpose factory)
    (test-rank factory)
    (test-mm factory)
    (test-mm-row-major factory)))
