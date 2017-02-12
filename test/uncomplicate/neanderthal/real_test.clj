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
  (with-release [b (.createDataSource acc (count s))]
    (reduce (fn [^long i ^double e] (do (.set acc b i e) (inc i))) 0 s)
    b))

(defn test-group [factory]
  (facts "Group methods."
         (with-release [x (vctr factory [1 2 3])
                        y (vctr factory [2 3 4])]

           (zero x) => (vctr factory [0 0 0])
           (identical? x (zero x)) => false)

         (with-release [a (ge factory 2 3 [1 2 3 4 5 6])
                        ac (ge factory 2 3 [0 0 0 0 0 0])]
           (zero a) => ac
           (identical? a (zero a)) => false)))

;; ============= Matrices and Vectors ======================================

(defn test-vctr-constructor [factory]
  (facts "Create a real vector."
         (vect? (vctr factory [1 2 3])) => true
         (vctr factory 1 2 3) => (vctr factory [1 2 3])
         (vctr factory 3) => (vctr factory [0 0 0])
         ;;(vctr factory (seq-to-buffer (data-accessor factory) [1 2 3]))
         ;;=> (vctr factory [1 2 3])
         (vctr factory nil) => (throws IllegalArgumentException)
         (dim (vctr factory [])) => 0
         (vctr factory 3) => (zero (vctr factory [1 2 3]))))

(defn test-vctr [factory]
  (facts "General vector functions."
         (dim (vctr factory [1 2 3])) => 3
         (dim (vctr factory [])) => 0

         (subvector (vctr factory 1 2 3 4 5 6) 1 3) => (vctr factory 2 3 4)
         (subvector (vctr factory 1 2 3) 2 3) => (throws IndexOutOfBoundsException)))

(defn test-vctr-entry [factory]
  (facts "Vectory entry."
         (entry (vctr factory [1 2 3]) 1) => 2.0
         (entry (vctr factory []) 0) => (throws IndexOutOfBoundsException)))

(defn test-vctr-entry! [factory]
  (facts "Vectory entry!."
         (entry (entry! (vctr factory [1 2 3]) 1 77.0) 1) => 77.0))

(defn test-vctr-bulk-entry! [factory]
  (facts "Vectory entry!."
         (sum (entry! (vctr factory [1 2 3]) 77.0)) => 231.0))

;;================ BLAS functions =========================================

(defn test-dot [factory]
  (facts "BLAS 1 dot."
         (dot (vctr factory [1 2 3]) (vctr factory [10 30 -10])) => 40.0
         (dot (vctr factory [1 2 3]) nil) => (throws IllegalArgumentException)
         (dot (vctr factory []) (vctr factory [])) => 0.0
         (dot (vctr factory [1]) (ge factory 3 3)) => (throws ClassCastException)
         (dot (vctr factory [1 2]) (vctr factory [1 2 3])) => (throws IllegalArgumentException)))

(defn test-nrm2 [factory]
  (facts "BLAS 1 nrm2."
         (nrm2 (vctr factory [1 2 3])) => (roughly (Math/sqrt 14))
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
         (with-release [x (vctr factory [1 2 3 4 5])
                        y (vctr factory [-1 -2 -3 -4 -5])]
           (do (rot! x y 1 0) [x y])
           => [(vctr factory 1 2 3 4 5) (vctr factory -1 -2 -3 -4 -5)]

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

           (rot! x y (/ (sqrt 3) 2.0) 0.5) => x
           (rot! x (vctr factory 1) 0.5 0.5) => (throws IllegalArgumentException)
           (rot! x y 300 0.3) => (throws IllegalArgumentException)
           (rot! x y 0.5 0.5) => (throws IllegalArgumentException))))

(defn test-rotg [factory]
  (facts "BLAS 1 rotg!"
         (with-release [x (vctr factory 1 2 3 4)]
           (rotg! x) => x)
         (rotg! (vctr factory 0 0 0 0)) => (vctr factory 0 0 1 0)
         (rotg! (vctr factory 0 2 0 0)) => (vctr factory 2 1 0 1)
         (nrm2 (axpy -1 (rotg! (vctr factory 6 -8 0 0))
                     (vctr factory -10 -5/3 -3/5 4/5)))
         => (roughly 0 1e-6)))

(defn test-rotm [factory]
  (facts "BLAS 1 rotm!"
         "TODO: Leave this for later since I am lacking good examples right now."))

(defn test-rotmg [factory]
  (facts "BLAS 1 rotmg!"
         "TODO: Leave this for later since I am lacking good examples right now."))

(defn test-vctr-swap [factory]
  (facts
   "BLAS 1 swap! vectors"
   (with-release [x (vctr factory [1 2 3])]
     (swp! x (vctr factory 3)) => x
     (swp! x (vctr factory [10 20 30])) => (vctr factory [10 20 30])
     (swp! x nil) => (throws IllegalArgumentException)
     (identical? (swp! x (vctr factory [10 20 30])) x) => true
     (swp! x (vctr factory [10 20])) => (throws IllegalArgumentException))))

(defn test-vctr-scal [factory]
  (facts "BLAS 1 scal!"
         (with-release [x (vctr factory [1 2 3])]()
           (identical? (scal! 3 x) x) => true)
         (scal! 3 (vctr factory [1 -2 3])) => (vctr factory [3 -6 9])
         (scal! 3 (vctr factory [])) => (vctr factory [])))

(defn test-vctr-copy [factory]
  (facts "BLAS 1 copy!"
         (with-release [y (vctr factory 3)]
           (identical? (copy! (vctr factory [1 2 3]) y) y) => true
           (identical? (copy y) y) => false
           (copy (vctr factory [1 2 3])) => y)

         (copy! (vctr factory [10 20 30]) (vctr factory [1 2 3]))
         => (vctr factory [10 20 30])

         (copy! (vctr factory [1 2 3]) nil) => (throws IllegalArgumentException)
         (copy! (vctr factory [10 20 30]) (vctr factory [1])) => (throws IllegalArgumentException)

         (copy (vctr factory 1 2)) => (vctr factory 1 2)))

(defn test-vctr-axpy [factory]
  (facts "BLAS 1 axpy!"
         (with-release [y (vctr factory [1 2 3])]
           (axpy! 2.0 (vctr factory [10 20 30]) y) => (vctr factory [21 42 63])
           (identical? (axpy! 3.0 (vctr factory [1 2 3]) y) y) => true
           (axpy! 2.0 (vctr factory [1 2]) y) => (throws IllegalArgumentException)

           (axpy! (vctr factory [10 20 30]) (vctr factory [1 2 3]))
           => (vctr factory [11 22 33]))

         (axpy! 2 (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7))
         => (vctr factory 25 33 41)

         (axpy! 2 (vctr factory 1 2 3) (vctr factory 1 2 3) (vctr factory 2 3 4) 4.0)
         => (throws IllegalArgumentException)

         (axpy! (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 1 2 3))
         => (axpy! 5 (vctr factory 1 2 3) (vctr factory 3))

         (axpy! 4 "af" (vctr factory 1 2 3) 3 "b" "c") => (throws IllegalArgumentException)

         (with-release [y (vctr factory [1 2 3])]
           (axpy! 2 (vctr factory 1 2 3) y
                  (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7)) => y))

  (facts "BLAS 1 axpby!"
         (with-release [y (vctr factory [1 2 3])]
           (axpby! 2.0 (vctr factory [10 20 30]) 2.0 y) => (vctr factory [22 44 66])
           (identical? (axpby! 3.0 (vctr factory [1 2 3]) 2.0 y) y) => true
           (axpby! 2.0 (vctr factory [1 2]) 2.0 y) => (throws IllegalArgumentException)

           (axpby! (vctr factory [10 20 30]) 2.0 (vctr factory [1 2 3]))
           => (vctr factory [12 24 36]))

         (axpby! 2 (vctr factory 1 2 3) 2 (vctr factory 1 2 3)
                (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7))
         => (vctr factory 26 35 44)

         (axpby! 2 (vctr factory 1 2 3) 2 (vctr factory 1 2 3) (vctr factory 2 3 4) 4.0)
         => (throws IllegalArgumentException)

         (axpby! (vctr factory 1 2 3) 2.0 (vctr factory 1 2 3)
                 (vctr factory 1 2 3) (vctr factory 1 2 3)
                 (vctr factory 1 2 3))
         => (axpy! 6 (vctr factory 1 2 3) (vctr factory 3))

         (axpy! 4 "af" (vctr factory 1 2 3) 3 "b" "c") => (throws IllegalArgumentException)

         (with-release [y (vctr factory [1 2 3])]
           (axpby! 2 (vctr factory 1 2 3) 2.0 y
                  (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7)) => y))

  (facts "BLAS1 axpy"
         (with-release [y (vctr factory [1 2 3])]
           (axpy 2.0 (vctr factory [10 20 30])) => (throws IllegalArgumentException)
           (identical? (axpy 3.0 (vctr factory [1 2 3]) y) y) => false

           (axpy (vctr factory 1 2 3) (vctr factory 2 3 4)
                 (vctr factory 3 4 5) (vctr factory 3))
           => (vctr factory 6 9 12)

           (axpy 2 (vctr factory 1 2 3) (vctr factory 2 3 4)) => (vctr factory 4 7 10))))

;; ================= Real General Matrix functions =============================

(defn test-ge-constructor [factory]
  (facts "Create a GE matrix."
         (with-release [a (ge factory 2 3 [1 2 3 4 5 6])]
           (ge factory 2 3 [1 2 3 4 5 6]) => a
           ;;(ge factory 2 3 (seq-to-buffer (data-accessor factory)
           ;;[1 2 3 4 5 6]))
           ;;=> a
           (ge factory 2 3 nil) => (zero a)
           ;;(ge factory 0 0 [])
           ;;=> (ge factory 0 0 (seq-to-buffer (data-accessor factory) []))
           )))

(defn test-ge [factory]
  (facts "GE Matrix methods."
         (with-release [a (ge factory 2 3 [1 2 3 4 5 6])]
           (mrows a) => 2
           (ncols a) => 3
           (row a 1) => (vctr factory [2 4 6])
           (col a 1) => (vctr factory [3 4]))

         (ncols (ge factory 2 3 [1 2 3 4 5 6])) => 3
         (ncols (ge factory 0 0 [])) => 0

         (mrows (ge factory 2 3)) => 2
         (mrows (ge factory 0 0)) => 0

         (row (ge factory 2 3 [1 2 3 4 5 6]) 1) => (vctr factory 2 4 6)
         (row (ge factory 2 3 [1 2 3 4 5 6]) 4) => (throws IndexOutOfBoundsException)

         (col (ge factory 2 3 [1 2 3 4 5 6]) 1) => (vctr factory 3 4)
         (col (ge factory 2 3 [1 2 3 4 5 6]) 4) => (throws IndexOutOfBoundsException)

         (submatrix (ge factory 3 4 (range 12)) 1 2 2 1) => (ge factory 2 1 [7 8])
         (submatrix (ge factory 3 4 (range 12)) 2 3) => (ge factory 2 3 [0 1 3 4 6 7])
         (submatrix (ge factory 3 4 (range 12)) 3 4) => (ge factory 3 4 (range 12))
         (submatrix (ge factory 3 4 (range 12)) 1 1 3 3) => (throws IndexOutOfBoundsException)

         (trans (ge factory 2 3 (range 6))) => (ge factory 3 2 [0 2 4 1 3 5] {:order row})))

(defn test-ge-entry [factory]
  (facts "GE matrix entry."
         (entry (ge factory 2 3 [1 2 3 4 5 6]) 0 1) => 3.0
         (entry (ge factory 0 0) 0 0)
         => (throws IndexOutOfBoundsException)
         (entry (ge factory 2 3 [1 2 3 4 5 6]) -1 2)
         => (throws IndexOutOfBoundsException)
         (entry (ge factory 2 3 [1 2 3 4 5 6]) 2 1)
         => (throws IndexOutOfBoundsException)))

(defn test-ge-entry! [factory]
  (facts "GE matrix entry!."
         (entry (entry! (ge factory 2 3 [1 2 3 4 5 6]) 0 1 88.0) 0 1) => 88.0))

(defn test-ge-bulk-entry! [factory]
  (facts "Bulk GE matrix entry!."
         (sum (row (entry! (ge factory 2 3 [1 2 3 4 5 6]) 88.0) 1)) => 264.0))

(defn test-ge-copy [factory]
  (facts "BLAS 1 copy! GE matrix"
         (with-release [a (ge factory 2 3 (range 6))
                        b (ge factory 2 3 (range 7 13))]
           (identical? (copy! a b) b) => true
           (copy! a b) => (ge factory 2 3 (range 6))
           (copy (ge factory 2 3 (range 6))) => a)

         (copy! (ge factory 2 3 [10 20 30 40 50 60]) (ge factory 2 3 [1 2 3 4 5 6]))
         => (ge factory 2 3 [10 20 30 40 50 60])

         (copy! (ge factory 2 3 [1 2 3 4 5 6]) nil) => (throws IllegalArgumentException)

         (copy! (ge factory 2 3 [10 20 30 40 50 60]) (ge factory 2 2))
         => (throws IllegalArgumentException)))

(defn test-ge-swap [factory]
  (facts
   "BLAS 1 swap! GE matrix"
   (with-release [a (ge factory 2 3 [1 2 3 4 5 6])]
     (swp! a (ge factory 2 3)) => a
     (swp! a (ge factory 2 3 [10 20 30 40 50 60])) => (ge factory 2 3 [10 20 30 40 50 60])
     (swp! a nil) => (throws IllegalArgumentException)
     (identical? (swp! a (ge factory 2 3 [10 20 30 40 50 60])) a) => true
     (swp! a (ge factory 1 2 [10 20])) => (throws IllegalArgumentException))))

(defn test-ge-scal [factory]
  (facts "BLAS 1 scal! GE matrix"
         (with-release [a (ge factory 2 3 [1 2 3 4 5 6])]
           (identical? (scal! 3 a) a) => true)
         (scal! 3 (ge factory 2 3 [1 -2 3 9 8 7])) => (ge factory 2 3 [3 -6 9 27 24 21])
         (scal! 3 (submatrix (ge factory 3 2 [1 2 3 4 5 6]) 1 1 1 1)) => (ge factory 1 1 [15])))

(defn test-ge-axpy [factory]
  (facts "BLAS 1 axpy! GE matrix"
         (with-release [a (ge factory 3 2 (range 6))]
           (axpy! 2.0 (ge factory 3 2 (range 0 60 10)) a) => (ge factory 3 2 (range 0 121 21))
           (identical? (axpy! 3.0 (ge factory 3 2 (range 6)) a) a) => true
           (axpy! 2.0 (ge factory 2 3 (range 1 7)) a) => (throws IllegalArgumentException)
           (axpy! (ge factory 1 3 [10 20 30]) (ge factory 1 3 [1 2 3])) => (ge factory 1 3 [11 22 33]))

         (axpy! 2 (ge factory 2 3 (range 6)) (ge factory 2 3 (range 6))
                (ge factory 2 3 (range 2 9)) 4.0 (ge factory 2 3 (range 8 14)))
         => (ge factory 2 3 (range 34 75 8))

         (axpy! 2 (ge factory 2 3 (range 6)) (ge factory 2 3 (range 6))
                (ge factory 2 3 (range 2 5)) 4.0)
         => (throws IllegalArgumentException)

         (axpy! (ge factory 2 3 (range 6)) (ge factory 2 3 (range 6))
                (ge factory 2 3 (range 6)) (ge factory 2 3 (range 6))
                (ge factory 2 3 (range 6)))
         => (axpy! 5 (ge factory 2 3 (range 6)) (ge factory 2 3))

         (axpy! 4 "af" (ge factory 2 3 (range 6)) 3 "b" "c") => (throws IllegalArgumentException)

         (with-release [a (ge factory 2 3 (range 6))]
           (axpy! 2 (ge factory 2 3 (range 6)) a (ge factory 2 3 (range 1 7))
                  4.0 (ge factory 2 3 (range 6))) => a)

         (with-release [a (ge factory 3 2 (range 6))]
           (axpy 2.0 (ge factory 2 3 (range 10 70 10))) => (throws IllegalArgumentException)
           (identical? (axpy 3.0 (ge factory 3 2 (range 1 7)) a) a) => false
           (axpy (ge factory 2 3 (range 1 7)) (ge factory 2 3 (range 2 8))
                 (ge factory 2 3 (range 3 9)) (ge factory 2 3))
           => (ge factory 2 3 (range 6 25 3))

           (axpy 2 (ge factory 2 3 (range 1 7)) (ge factory 2 3 (range 2 8)))
           => (ge factory 2 3 (range 4 23 3)))))

;; ====================== BLAS 2 ===============================

(defn test-ge-mv [factory]
  (facts "BLAS 2 GE mv!"
         (mv! 2.0 (ge factory 3 2 [1 2 3 4 5 6])
              (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4])) => (throws IllegalArgumentException)

         (with-release [y (vctr factory [1 2 3])]
           (identical? (mv! 2 (ge factory 3 2 [1 2 3 4 5 6]) (vctr factory 1 2) 3 y) y))
         => true

         (mv! (ge factory 2 3 [1 10 2 20 3 30])
              (vctr factory 7 0 4) (vctr factory 0 0)) => (vctr factory 19 190)

         (mv! 2.0 (ge factory 2 3 [1 2 3 4 5 6])
              (vctr factory 1 2 3) 3.0 (vctr factory 1 2)) => (vctr factory 47 62)

         (mv! 2.0 (ge factory 2 3 [1 2 3 4 5 6]) (vctr factory 1 2 3) 0.0 (vctr factory 0 0))
         => (mv 2.0 (ge factory 2 3 [1 2 3 4 5 6]) (vctr factory 1 2 3))

         (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
              (vctr factory [1 3 5]) 3.0 (vctr factory [2 3]))
         => (vctr factory 272 293)

         (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
              (vctr factory [1 3 5]) 3.0 (col (ge factory 2 3 (range 6)) 1))
         => (vctr factory 272 293)

         (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
              (row (ge factory 2 3 (range 6)) 1) 3.0 (col (ge factory 2 3 (range 6)) 1))
         => (vctr factory 272 293)

         (mv! (ge factory 2 3 [1 3 5 2 4 6] {:order :row}) (vctr factory 1 2 3) (vctr factory 0 0))
         => (mv! (ge factory 2 3 [1 2 3 4 5 6] {:order :column}) (vctr factory 1 2 3) (vctr factory 0 0))

         (mv 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:order :row}) (vctr factory 1 2 3) (vctr factory 0 0))
         => (vctr factory 22 28)))

(defn test-rank [factory]
  (facts "BLAS 2 rank!"
         (with-release [a (ge factory 2 3)]
           (identical? (rank! 2.0 (vctr factory 1 2) (vctr factory 1 2 3) a) a))
         => true

         (rank! (vctr factory 3 2 1 4) (vctr factory 1 2 3)
                (ge factory 4 3 [1 2 3 4 2 2 2 2 3 4 2 1]))
         => (ge factory 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

         (rank! (vctr factory 1 2) (vctr factory 1 2 3) (ge factory 2 2 [1 2 3 5]))
         => (throws IllegalArgumentException)))

;; ====================== BLAS 3 ================================

(defn test-ge-mm [factory]
  (facts "BLAS 3 GE mm!"
         (mm! 2.0 (ge factory 3 2 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws IllegalArgumentException)

         (mm! 2.0 (ge factory 3 2 [1 2 3 4 5 6]) (ge factory 2 3 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws IllegalArgumentException)

         (with-release [c (ge factory 2 2 [1 2 3 4])]
           (identical? (mm! 2.0 (ge factory 2 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6]) 3.0 c) c))
         => true

         (mm! 2.0 (ge factory 2 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [47 62 107 140])

         (mm! 2.0 (ge factory 3 5 (take 15 (repeat 1))) (ge factory 5 3 (take 15 (repeat 1)))
              3.0 (ge factory 3 3 (take 9 (repeat 0)))) => (ge factory 3 3 (take 9 (repeat 10)))

         (mm! 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:order :row}) (ge factory 3 2 [1 4 2 5 3 6] {:order row})
              3.0  (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [49 66 97 128])

         (mm 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:order :row}) (ge factory 3 2 [1 4 2 5 3 6] {:order row})
             3.0  (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [49 66 97 128])

         (mm (ge factory 2 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6]))
         => (ge factory 2 2 (list 22.0 28.0 49.0 64.0))))

;; ====================== TR Matrix ============================

(defn test-tr-constructor [factory]
  (facts "Create a triangular matrix."
         (with-release [a (tr factory 3 (range 6))]
           (subtriangle (ge factory 3 3 [0 1 2 0 3 4 0 0 5])) => a
           ;;(ge factory 2 3 (seq-to-buffer (data-accessor factory)
           ;;[1 2 3 4 5 6]))
           ;;=> a
           (tr factory 3 nil) => (zero a)
           ;;(ge factory 0 0 [])
           ;;=> (ge factory 0 0 (seq-to-buffer (data-accessor factory) []))
           )))

(defn test-tr [factory]
  (facts "Triangular Matrix methods."
         (with-release [a-upper (tr factory 3 (range 15) {:uplo :upper})
                        a-lower (tr factory 3 (range 15) {:uplo :lower})]
           (= 3 (mrows a-upper) (ncols a-lower)) => true
           (row a-upper 0) => (vctr factory [0 1 3])
           (row a-upper 2) => (vctr factory [5])
           (col a-upper 0) => (vctr factory [0])
           (col a-upper 2) => (vctr factory [3 4 5])
           (row a-lower 0) => (vctr factory [0])
           (row a-lower 2) => (vctr factory [2 4 5])
           (col a-lower 0) => (vctr factory [0 1 2])
           (col a-lower 2) => (vctr factory [5]))))

(defn test-tr-entry [factory]
  (facts "TR Matrix entry."
         (with-release [a-upper (tr factory 3 (range 6) {:uplo :upper})
                        b-diag (tr factory 3 (range 6) {:diag :unit})]
           (entry a-upper 0 1) => 1.0
           (entry a-upper 1 0) => 0.0
           (entry a-upper 1 1) => 2.0
           (entry a-upper 2 0) => 0.0
           (entry a-upper 0 2) => 3.0
           (entry a-upper -1 2) => (throws IndexOutOfBoundsException)
           (entry a-upper 3 2) => (throws IndexOutOfBoundsException)
           (entry b-diag 2 2) => 1.0
           (entry b-diag 2 1) => 2.0)))

(defn test-tr-entry! [factory]
  (facts "TR matrix entry!."
         (with-release [a (tr factory 3 [1 2 3 4 5 6] {:diag :unit :uplo :upper})]
           (entry (entry! a 0 1 88.0) 0 1) => 88.0
           (entry (entry! a 1 0 3.0) 0 1) => (throws IndexOutOfBoundsException)
           (entry (entry! a 1 0 4.0) 1 1) => (throws IndexOutOfBoundsException))))

(defn test-tr-bulk-entry! [factory]
  (facts "Bulk TR matrix entry!."
         (sum (row (entry! (tr factory 3 [1 2 3 4 5 6]) 33.0) 1)) => 66.0
         (sum (row (entry! (tr factory 3 [1 2 3 4 5 6] {:diag :unit :uplo :upper}) 22.0) 0)) => 44.0))

(defn test-tr-copy [factory]
  (facts "BLAS 1 copy! TR matrix"
         (with-release [a (tr factory 3)]
           (identical? (copy a) a) => false
           (identical? (copy! (tr factory 3 [1 2 3 4 5 6]) a) a) => true
           (copy (tr factory 3 [1 2 3 4 5 6])) => a)

         (copy! (tr factory 3 [10 20 30 40 50 60]) (tr factory 3 [1 2 3 4 5 6]))
         => (tr factory 3 [10 20 30 40 50 60])

         (copy! (tr factory 3 [1 2 3 4 5 6]) nil) => (throws IllegalArgumentException)
         (copy! (tr factory 3 [10 20 30 40 50 60]) (tr factory 2)) => (throws IllegalArgumentException)))

(defn test-tr-swap [factory]
   (facts
    "BLAS 1 swap! TR matrix"
    (with-release [a (tr factory 3 [1 2 3 4 5 6])]
      (identical? (swp! a (tr factory 3)) a) => true
      (swp! a (tr factory 3 [10 20 30 40 50 60])) => (tr factory 3 [10 20 30 40 50 60])
      (swp! a nil) => (throws IllegalArgumentException)
      (identical? (swp! a (tr factory 3 [10 20 30 40 50 60])) a) => true
      (swp! a (tr factory 2 [10 20])) => (throws IllegalArgumentException))))

(defn test-tr-scal [factory]
  (facts "BLAS 1 scal! TR matrix"
         (with-release [a (tr factory 3 [1 2 3 4 5 6])]
           (identical? (scal! 3 a) a) => true)
         (scal! 3 (tr factory 3 [1 -2 3 9 8 7])) => (tr factory 3 [3 -6 9 27 24 21])))

(defn test-tr-axpy [factory]
  (facts "BLAS 1 axpy! TR matrix"
         (with-release [a (tr factory 3 (range 6))]

           (axpy! 2.0 (tr factory 3 (range 0 60 10)) a) => (tr factory 3 (range 0 121 21))
           (identical? (axpy! 3.0 (tr factory 3 (range 6)) a) a) => true
           (axpy! 2.0 (tr factory 2 (range 1 7)) a) => (throws IllegalArgumentException)

           (axpy! (tr factory 2 [10 20 30]) (tr factory 2 [1 2 3])) => (tr factory 2 [11 22 33]))

         (axpy! 2 (tr factory 3 (range 6)) (tr factory 3 (range 6))
                (tr factory 3 (range 2 9)) 4.0 (tr factory 3 (range 8 14)))
         => (tr factory 3 (range 34 75 8))

         (axpy! 2 (tr factory 3 (range 6)) (tr factory 3 (range 6))
                (tr factory 2 3 (range 2 5)) 4.0) => (throws IllegalArgumentException)

         (axpy! (tr factory 3 (range 6)) (tr factory 3 (range 6))
                (tr factory 3 (range 6)) (tr factory 3 (range 6))
                (tr factory 3 (range 6)))
         => (axpy! 5 (tr factory 3 (range 6)) (tr factory 3))

         (axpy! 4 "af" (tr factory 3 (range 6)) 3 "b" "c") => (throws IllegalArgumentException)

         (with-release [a (tr factory 3 (range 6))]
           (axpy! 2 (tr factory 3 (range 6)) a
                  (tr factory 3 (range 1 7)) 4.0 (tr factory 3 (range 6))) => a)

         (with-release [a (tr factory 3 (range 6))]
           (axpy 2.0 (tr factory 3 (range 10 70 10))) => (throws IllegalArgumentException)
           (identical? (axpy 3.0 (tr factory 3 (range 1 7)) a) a) => false

           (axpy (tr factory 3 (range 1 7)) (tr factory 3 (range 2 8))
                 (tr factory 3 (range 3 9)) (tr factory 3))
           => (tr factory 3 (range 6 25 3))

           (axpy 2 (tr factory 3 (range 1 7)) (tr factory 3 (range 2 8))) => (tr factory 3 (range 4 23 3)))))

(defn test-tr-mv [factory]
  (facts "BLAS 2 TR mv!"
         (mv! 2.0 (tr factory 2 [1 2 3 4])
              (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4])) => (throws IllegalArgumentException)

         (with-release [y (vctr factory [1 2 3])]
           (identical? (mv! (tr factory 3 [1 2 3 4 5 6]) y) y)) => true

         (mv! (tr factory 3 [1 10 2 20 3 30] {:order :row}) (vctr factory 7 0 4))
         => (vctr factory 7 70 260)

         (mv! (tr factory 3 [1 2 3 4 5 6] {:diag :unit :uplo :upper})
              (vctr factory 1 2 3)) => (vctr factory 9 11 3)

         (mv! (tr factory 3 [1 2 3 4 5 6]) (vctr factory 1 2 3))
         => (mv (tr factory 3 [1 2 3 4 5 6]) (vctr factory 1 2 3))

         (mv! (tr factory 3 [1 2 4 3 5 6] {:order :row}) (vctr factory 1 2 3))
         => (mv! (tr factory 3 [1 2 3 4 5 6] {:order :column}) (vctr factory 1 2 3))))

(defn test-tr-mm [factory]
  (facts "BLAS 3 TR mm!"
         (mm! 2.0 (tr factory 3 [1 2 3 4 5 6]) (tr factory 3 [1 2 3 4 5 6])
              3.0 (tr factory 3 [1 2 3 4 5 6]))=> (throws IllegalArgumentException)

         (mm! 2.0 (tr factory 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws IllegalArgumentException)

         (with-release [c (ge factory 2 2 [1 2 3 4])]
           (identical? (mm! 2.0 (tr factory 2 [1 2 3 4]) c) c)) => true

         (mm! 2.0 (ge factory 2 3 (range 6)) (tr factory 3 (range 6)))
         => (mm! 2.0 (ge factory 2 3 (range 6)) (ge factory 3 3 [0 1 2 0 3 4 0 0 5]) 0.0 (ge factory 2 3))

         (mm (tr factory 3 (range 6)) (ge factory 3 2 (range 6)))
         => (ge factory 3 2 [0 3 14 0 15 47])

         (mm (ge factory 2 3 (range 6)) (tr factory 3 (range 6)))
         => (ge factory 2 3 [10 13 22 29 20 25])

         (mm 2.0 (tr factory 3 [1 2 3 4 5 6]) (tr factory 3 [1 2 3 4 5 6])
             3.0 (tr factory 3 [1 2 3 4 5 6])) => (throws IllegalArgumentException)))

;; =========================================================================
(defn test-all [factory]
  (test-group factory)
  (test-vctr-constructor factory)
  (test-vctr factory)
  (test-vctr-bulk-entry! factory)
  (test-dot factory)
  (test-nrm2 factory)
  (test-asum factory)
  (test-sum factory)
  (test-iamax factory)
  (test-vctr-swap factory)
  (test-vctr-copy factory)
  (test-vctr-scal factory)
  (test-vctr-axpy factory)
  (test-ge-constructor factory)
  (test-ge factory)
  (test-ge-bulk-entry! factory)
  (test-ge-swap factory)
  (test-ge-copy factory)
  (test-ge-scal factory)
  (test-ge-axpy factory)
  (test-ge-mv factory)
  (test-rank factory)
  (test-ge-mm factory)
  (test-tr factory)
  (test-tr-constructor factory)
  (test-tr-copy factory)
  (test-tr-swap factory)
  (test-tr-scal factory)
  (test-tr-axpy factory)
  (test-tr-mv factory)
  (test-tr-mm factory))

(defn test-host [factory]
  (test-rot factory)
  (test-rotg factory)
  (test-rotm factory)
  (test-rotmg factory)
  (test-vctr-entry factory)
  (test-vctr-entry! factory)
  (test-ge-entry factory)
  (test-ge-entry! factory)
  (test-tr-entry factory)
  (test-tr-entry! factory)
  (test-tr-bulk-entry! factory))
