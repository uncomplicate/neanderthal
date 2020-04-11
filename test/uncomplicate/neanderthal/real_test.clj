;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.real-test
  (:require [midje.sweet :refer [facts throws => roughly truthy]]
            [uncomplicate.commons.core :refer [release with-release let-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.fluokitten.protocols :refer [id]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [linalg :refer :all]
             [auxil :refer :all]
             [math :refer :all]
             [vect-math :refer [linear-frac!]]
             [real :refer [ls-residual]]]
            [uncomplicate.neanderthal.internal.api :refer [data-accessor index-factory]])
  (:import clojure.lang.ExceptionInfo
           [uncomplicate.neanderthal.internal.api MatrixImplementation]))

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

(defn test-diag-equal [factory]
  (facts "Test equals on diagonal matrix types"
         (with-release
           [gd-a (gd factory 5 [100 200 300 400 500])
            gd-copy (gd factory gd-a)
            gd-zero (gd factory 5)
            gd-diff (gd factory 5 [100 200 300 600 500])
            st-a (st factory 5 [100 200 300 400 500 5 6 7 8])
            st-copy (st factory st-a)
            st-zero (st factory 5)
            st-diff (st factory 5 [100 200 300 400 500 5 6 9 8])
            gt-a (gt factory 5 [100 200 300 400 500 6 7 8 9 10 11 12 13])
            gt-copy (gt factory 5 gt-a)
            gt-zero (gt factory 5)
            gt-diff (gt factory 5 [100 200 300 400 500 6 7 8 9 10 11 14 13])
            dt-a (dt factory 5 [100 200 300 400 500 6 7 8 9 10 11 12 13])
            dt-copy (dt factory 5 dt-a)
            dt-zero (dt factory 5)
            dt-diff (dt factory 5 [100 200 300 400 500 6 7 8 9 10 11 14 13])]

           (= gd-a gd-a) => true
           (= gd-a gd-copy) => true
           (= gd-a gd-zero) => false
           (= gd-a gd-diff) => false

           (= st-a st-a) => true
           (= st-a st-copy) => true
           (= st-a st-zero) => false
           (= st-a st-diff) => false

           (= gt-a gt-a) => true
           (= gt-a gt-copy) => true
           (= gt-a gt-zero) => false
           (= gt-a gt-diff) => false

           (= dt-a dt-a) => true
           (= dt-a dt-copy) => true
           (= dt-a dt-zero) => false
           (= dt-a dt-diff) => false

           (= gt-a dt-a) => true
           (= gt-a dt-copy) => true
           (= gt-a dt-zero) => false
           (= gt-a dt-diff) => false

           (= gd-zero st-zero) => false
           (= gd-zero gt-zero) => false
           (= gd-zero dt-zero) => false
           (= st-zero gt-zero) => false
           (= st-zero dt-zero) => false
           (= gt-zero dt-zero) => true

           (= (id gd-a) (id st-a) (id gt-a) (id dt-a) (id gt-a)) => true)))

;; ============= Matrices and Vectors ======================================

(defn test-vctr-constructor [factory]
  (facts "Create a real vector."
         (vctr? (vctr factory [1 2 3])) => true
         (vctr factory 1 2 3) => (vctr factory [1 2 3])
         (vctr factory 3) => (vctr factory [0 0 0])
         (view (vctr factory 1 2 3)) => (vctr factory 1 2 3)
         (view-vctr (vctr factory (range 8))) => (vctr factory (range 8))
         (view-vctr (vctr factory (range 8)) 3) => (vctr factory [0 3 6])
         (view-ge (vctr factory 1 2 3 4)) => (ge factory 4 1 [1 2 3 4])
         (vctr factory nil) => (throws IllegalArgumentException)
         (dim (vctr factory [])) => 0
         (vctr factory 3) => (zero (vctr factory [1 2 3]))))

(defn test-vctr [factory]
  (facts "General vector functions."
         (dim (vctr factory [1 2 3])) => 3
         (dim (vctr factory [])) => 0

         (subvector (vctr factory 1 2 3 4 5 6) 1 3) => (vctr factory 2 3 4)
         (subvector (vctr factory 1 2 3) 2 3) => (throws ExceptionInfo)))

(defn test-vctr-entry [factory]
  (facts "Vector entry."
         (entry (vctr factory [1 2 3]) 1) => 2.0
         (entry (vctr factory []) 0) => (throws ExceptionInfo)))

(defn test-vctr-entry! [factory]
  (facts "Vector entry!."
         (entry (entry! (vctr factory [1 2 3]) 1 77.0) 1) => 77.0))

(defn test-vctr-bulk-entry! [factory]
  (facts "Vector entry!."
         (sum (entry! (vctr factory [1 2 3]) 77.0)) => 231.0))

(defn ^:private val+
  (^double [^double val]
   (inc val)))

(defn ^:private val-ind+
  (^double [^long i ^double val]
   (+ (double i) val))
  (^double [^long i ^long j ^double val]
   (* (double j) (+ (double i) val))))

(defn test-vctr-alter! [factory]
  (facts "Vector alter!."
         (entry (alter! (vctr factory [1 2 3]) 1 val+) 1) => 3.0
         (alter! (vctr factory [1 2 3]) val-ind+) => (vctr factory [1 3 5])))

(defn test-vctr-int-entry [factory]
  (facts "Vector entry."
         (entry (vctr factory [1 2 3 4]) 1) => 2
         (entry (vctr factory []) 0) => (throws ExceptionInfo)))

(defn test-vctr-int-entry! [factory]
  (facts "Vector entry!."
         (entry (entry! (vctr factory [1 2 3 4]) 1 77) 1) => 77))

(defn test-vctr-int-alter! [factory]
  (facts "Vector alter!."
         (entry (alter! (vctr factory [1 2 3 4]) 1
                        (fn ^long [^long val] (inc val))) 1) => 3
         (alter! (vctr factory [1 2 3 4])
                 (fn ^long [^long i ^long val] (long (+ i val)))) => (vctr factory [1 3 5 7])))


;;================ BLAS functions =========================================

(defn test-vctr-dot [factory]
  (facts "BLAS 1 dot."
         (dot (vctr factory [1 2 3]) (vctr factory [10 30 -10])) => 40.0
         (dot (vctr factory [1 2 3]) nil) => (throws ExceptionInfo)
         (dot (vctr factory []) (vctr factory [])) => 0.0
         (dot (vctr factory [1]) (ge factory 3 3)) => (throws ExceptionInfo)
         (dot (vctr factory [1 2]) (vctr factory [1 2 3])) => (throws ExceptionInfo)))

(defn test-sum [factory]
  (facts "BLAS Plus 1 sum."
         (sum (vctr factory [1 2 -5]))  => -2.0
         (sum (vctr factory [])) => 0.0))

(defn test-sum-asum [factory]
  (with-release [x (vctr factory (range 1 100 0.001))
                 y (vctr factory [1 2 4 9])
                 z (entry! (vctr factory 1000000) 1.0)]
    (sum x) => (asum x)
    (sum y) => (asum y)
    (sum x) => (asum x)))

(defn test-iamax [factory]
  (facts "BLAS 1 iamax"
         (iamax (vctr factory [1 2 7 7 6 2 -12 10])) => 6
         (iamax (vctr factory [])) => 0
         (iamax (vctr factory [4 6 7 3])) => 2))

(defn test-iamin [factory]
  (facts "BLAS 1 iamax"
         (iamin (vctr factory [1 2 7 7 6 2 -12 10])) => 0
         (iamin (vctr factory [])) => 0
         (iamin (vctr factory [4 6 7 3])) => 3))

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
           (rot! x (vctr factory 1) 0.5 0.5) => (throws ExceptionInfo)
           (rot! x y 300 0.3) => (throws ExceptionInfo)
           (rot! x y 0.5 0.5) => (throws ExceptionInfo))))

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

(defn test-vctr-nrm2 [factory]
  (facts "BLAS 1 vector nrm2."
         (nrm2 (vctr factory [1 2 3])) => (roughly (Math/sqrt 14))
         (nrm2 (vctr factory [])) => 0.0))

(defn test-vctr-asum [factory]
  (facts "BLAS 1 vector asum."
         (asum (vctr factory [1 2 -3]))  => 6.0
         (asum (vctr factory [])) => 0.0))

(defn test-vctr-amax [factory]
  (facts "BLAS 1 vector amax."
         (amax (vctr factory [1 2 3 -4])) => 4.0
         (amax (vctr factory [])) => 0.0))

(defn test-vctr-swap [factory]
  (facts
   "BLAS 1 vectors swap!"
   (with-release [x (vctr factory [1 2 3 4])]
     (swp! x (vctr factory 4)) => x
     (swp! x (vctr factory [10 20 30 40])) => (vctr factory [10 20 30 40])
     (swp! x nil) => (throws ExceptionInfo)
     (identical? (swp! x (vctr factory [10 20 30 40])) x) => true
     (swp! x (vctr factory [10 20])) => (throws ExceptionInfo))))

(defn test-vctr-scal [factory]
  (facts "BLAS 1 vector scal!"
         (with-release [x (vctr factory [1 2 3])]()
           (identical? (scal! 3 x) x) => true)
         (scal! 3 (vctr factory [1 -2 3])) => (vctr factory [3 -6 9])
         (scal! 3 (vctr factory [])) => (vctr factory [])))

(defn test-vctr-copy [factory]
  (facts "BLAS 1 vector copy!"
         (with-release [y (vctr factory 4)]
           (identical? (copy! (vctr factory [1 2 3 4]) y) y) => true
           (identical? (copy y) y) => false
           (copy (vctr factory [1 2 3 4])) => y)

         (copy! (vctr factory [10 20 30 40]) (vctr factory [1 2 3 4]))
         => (vctr factory [10 20 30 40])

         (copy! (vctr factory [1 2 3 4]) nil) => (throws ExceptionInfo)
         (copy! (vctr factory [10 20 30 40]) (vctr factory [1])) => (throws ExceptionInfo)

         (copy (vctr factory 1 2 3 4)) => (vctr factory 1 2 3 4)))

(defn test-vctr-axpy [factory]
  (facts "BLAS 1 vector axpy!"
         (with-release [y (vctr factory [1 2 3])]
           (axpy! 2.0 (vctr factory [10 20 30]) y) => (vctr factory [21 42 63])
           (identical? (axpy! 3.0 (vctr factory [1 2 3]) y) y) => true
           (axpy! 2.0 (vctr factory [1 2]) y) => (throws ExceptionInfo)

           (axpy! (vctr factory [10 20 30]) (vctr factory [1 2 3]))
           => (vctr factory [11 22 33]))

         (axpy! 2 (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7))
         => (vctr factory 25 33 41)

         (axpy! 2 (vctr factory 1 2 3) (vctr factory 1 2 3) (vctr factory 2 3 4) 4.0)
         => (throws ExceptionInfo)

         (axpy! (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 1 2 3) (vctr factory 1 2 3)
                (vctr factory 1 2 3))
         => (axpy! 5 (vctr factory 1 2 3) (vctr factory 3))

         (axpy! 4 "af" (vctr factory 1 2 3) 3 "b" "c") => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3])]
           (axpy! 2 (vctr factory 1 2 3) y
                  (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7)) => y))

  (facts "BLAS 1 vector axpby!"
         (with-release [y (vctr factory [1 2 3])]
           (axpby! 2.0 (vctr factory [10 20 30]) 2.0 y) => (vctr factory [22 44 66])
           (identical? (axpby! 3.0 (vctr factory [1 2 3]) 2.0 y) y) => true
           (axpby! 2.0 (vctr factory [1 2]) 2.0 y) => (throws ExceptionInfo)

           (axpby! (vctr factory [10 20 30]) 2.0 (vctr factory [1 2 3]))
           => (vctr factory [12 24 36]))

         (axpby! 2 (vctr factory 1 2 3) 2 (vctr factory 1 2 3)
                 (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7))
         => (vctr factory 26 35 44)

         (axpby! 2 (vctr factory 1 2 3) 2 (vctr factory 1 2 3) (vctr factory 2 3 4) 4.0)
         => (throws ExceptionInfo)

         (axpby! (vctr factory 1 2 3) 2.0 (vctr factory 1 2 3)
                 (vctr factory 1 2 3) (vctr factory 1 2 3)
                 (vctr factory 1 2 3))
         => (axpy! 6 (vctr factory 1 2 3) (vctr factory 3))

         (axpy! 4 "af" (vctr factory 1 2 3) 3 "b" "c") => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3])]
           (axpby! 2 (vctr factory 1 2 3) 2.0 y
                   (vctr factory 2 3 4) 4.0 (vctr factory 5 6 7)) => y))

  (facts "BLAS1 vector axpy"
         (with-release [y (vctr factory [1 2 3])]
           (axpy 2.0 (vctr factory [10 20 30])) => (throws ExceptionInfo)
           (identical? (axpy 3.0 (vctr factory [1 2 3]) y) y) => false

           (axpy (vctr factory 1 2 3) (vctr factory 2 3 4)
                 (vctr factory 3 4 5) (vctr factory 3))
           => (vctr factory 6 9 12)

           (axpy 2 (vctr factory 1 2 3) (vctr factory 2 3 4)) => (vctr factory 4 7 10))))

;; ================= Real General Matrix functions =============================

(defn test-ge-constructor [factory]
  (facts "Create a GE matrix."
         (with-release [a (ge factory 2 4 [1 2 5 6 9 10 13 14])
                        a1 (ge factory 2 4 [1 2 5 6 9 10 13 14])
                        a2 (ge factory 4 2 [1 2 5 6 9 10 13 14])
                        v (vctr factory [1 2 5 6 9 10 13 14])
                        b (tr factory 2 [1 2 6])
                        c (ge factory 2 7 (range 1 15))]
           a1 => a
           (ge factory 2 4 nil) => (zero a)
           (ge factory [[1 2] [5 6] [9 10] [13 14]]) => a
           (view a) => a
           (view-vctr a) => v
           (view-ge a) => a
           (view-ge c 2) => a
           (view-ge a2 2 4) => a
           (view-tr a) => b)))

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
         (row (ge factory 2 3 [1 2 3 4 5 6]) 4) => (throws ExceptionInfo)

         (col (ge factory 2 3 [1 2 3 4 5 6]) 1) => (vctr factory 3 4)
         (col (ge factory 2 3 [1 2 3 4 5 6]) 4) => (throws ExceptionInfo)

         (dia (ge factory 2 3 [1 2 3 4 5 6])) => (vctr factory 1 4)

         (submatrix (ge factory 3 4 (range 12)) 1 2 2 1) => (ge factory 2 1 [7 8])
         (submatrix (ge factory 3 4 (range 12)) 2 3) => (ge factory 2 3 [0 1 3 4 6 7])
         (submatrix (ge factory 3 4 (range 12)) 3 4) => (ge factory 3 4 (range 12))
         (submatrix (ge factory 3 4 (range 12)) 1 1 3 3) => (throws ExceptionInfo)

         (trans (ge factory 2 3 (range 6))) => (ge factory 3 2 (range 6) {:layout :row})))

(defn test-ge-entry [factory]
  (facts "GE matrix entry."
         (entry (ge factory 2 3 [1 2 3 4 5 6]) 0 1) => 3.0
         (entry (ge factory 0 0) 0 0) => (throws ExceptionInfo)
         (entry (ge factory 2 3 [1 2 3 4 5 6]) -1 2) => (throws ExceptionInfo)
         (entry (ge factory 2 3 [1 2 3 4 5 6]) 2 1) => (throws ExceptionInfo)))

(defn test-ge-entry! [factory]
  (facts "GE matrix entry!."
         (entry (entry! (ge factory 2 3 [1 2 3 4 5 6]) 0 1 88.0) 0 1) => 88.0))

(defn test-ge-bulk-entry! [factory]
  (facts "Bulk GE matrix entry!."
         (sum (row (entry! (ge factory 2 3 [1 2 3 4 5 6]) 88.0) 1)) => 264.0))

(defn test-ge-alter! [factory]
  (facts "GE alter!."
         (entry (alter! (ge factory 2 3 [1 2 3 4 5 6] {:layout :row}) 1 1 val+) 1 1) => 6.0
         (alter! (ge factory 2 3 [1 2 3 4 5 6]) val-ind+) => (ge factory 2 3 [0 0 3 5 10 14])
         (alter! (ge factory 2 3 [1 2 3 4 5 6] {:layout :row}) val-ind+)
         => (ge factory 2 3 [0 2 6 0 6 14] {:layout :row})))

(defn test-ge-dot [factory]
  (facts "BLAS 1 GE dot"
         (with-release [a (ge factory 2 3 (range -3 3))
                        b (ge factory 2 3 [1 2 3 4 5 6])
                        zero-point (ge factory 0 0 [])]
           (sqrt (dot a a)) => (roughly (nrm2 a))
           (dot a b) => (dot (view-vctr a) (view-vctr b))
           (dot zero-point zero-point) => 0.0)))

(defn test-ge-dot-strided [factory]
  (facts "BLAS 1 GE dot"
         (with-release [a (ge factory 2 3 (range -3 3))
                        b (ge factory 2 3 [1 2 3 4 5 6])
                        c (ge factory 3 4 (repeat 1))
                        d (ge factory 2 3 {:layout :row})
                        zero-point (ge factory 0 0 [])]
           (dot a (copy! b (submatrix c 1 1 2 3))) => (dot a b)
           (dot a (copy! b d)) => (dot a b))))

(defn test-ge-nrm2 [factory]
  (facts "BLAS 1 GE nrm2."
         (nrm2 (ge factory 2 3 (range -3 3))) => (roughly (Math/sqrt 19))
         (nrm2 (ge factory 0 0 [])) => 0.0))

(defn test-ge-asum [factory]
  (facts "BLAS 1 GE asum."
         (asum (ge factory 2 3 (range -3 3))) => 9.0
         (asum (ge factory 0 0 [])) => 0.0))

(defn test-ge-sum [factory]
  (facts "BLAS 1 GE sum."
         (sum (ge factory 2 3 (range -3 3))) => -3.0
         (sum (ge factory 0 0 [])) => 0.0))

(defn test-ge-sum-asum [factory]
  (with-release [a (ge factory 1234 32 (range 1 1000 0.001))
                 b (ge factory 2 3 [1 2 4 9 19 89])
                 c (entry! (ge factory 32 1234 (range 1 1000 0.001)) 1.0)]
    (sum a) => (asum a)
    (sum b) => (asum b)
    (sum c) => (asum c)))

(defn test-ge-amax [factory]
  (facts "BLAS 1 GE amax."
         (amax (ge factory 2 3 [1 2 3 -7.1 -3 1])) => (roughly 7.1)
         (amax (ge factory 0 0 [])) => 0.0))

(defn test-ge-trans! [factory]
  (facts "BLAS 1 GE trans!."
         (trans! (ge factory 2 3 (range 6))) => (ge factory 2 3 [0 2 4 1 3 5])
         (trans! (submatrix (ge factory 4 3 (range 12)) 1 0 2 3)) => (throws ExceptionInfo)))

(defn test-ge-copy [factory]
  (facts "BLAS 1 copy! GE matrix"
         (with-release [a (ge factory 2 3 (range 6))
                        b (ge factory 2 3 (range 7 13))
                        b-row (ge factory 2 3 {:layout :row})]
           (identical? (copy! a b) b) => true
           (copy! a b) => (ge factory 2 3 (range 6))
           (copy (ge factory 2 3 (range 6))) => a
           (copy! a b-row) => a)

         (copy! (ge factory 2 3 [10 20 30 40 50 60]) (ge factory 2 3 [1 2 3 4 5 6]))
         => (ge factory 2 3 [10 20 30 40 50 60])

         (copy! (ge factory 2 3 [1 2 3 4 5 6]) nil) => (throws ExceptionInfo)

         (copy! (ge factory 2 3 [10 20 30 40 50 60]) (ge factory 2 2)) => (throws ExceptionInfo)))

(defn test-ge-swap [factory]
  (facts
   "BLAS 1 swap! GE matrix"
   (with-release [a (ge factory 2 3 [1 2 3 4 5 6])
                  b (ge factory 2 3 (range 7 13))
                  b-row (ge factory 2 3 [7 9 11 8 10 12] {:layout :row})]
     (swp! a (ge factory 2 3)) => a
     (swp! a (ge factory 2 3 [10 20 30 40 50 60])) => (ge factory 2 3 [10 20 30 40 50 60])
     (swp! a b-row) => b

     (swp! a nil) => (throws ExceptionInfo)
     (identical? (swp! a (ge factory 2 3 [10 20 30 40 50 60])) a) => true
     (swp! a (ge factory 1 2 [10 20])) => (throws ExceptionInfo))))

(defn test-ge-scal [factory]
  (facts "BLAS 1 scal! GE matrix"
         (with-release [a (ge factory 2 3 [1 2 3 4 5 6])]
           (identical? (scal! 3 a) a) => true)
         (scal! 3 (ge factory 2 3 [1 -2 3 9 8 7])) => (ge factory 2 3 [3 -6 9 27 24 21])
         (scal! 3 (submatrix (ge factory 3 2 [1 2 3 4 5 6]) 1 1 1 1)) => (ge factory 1 1 [15])))

(defn test-ge-axpy [factory]
  (facts "BLAS 1 axpy! GE matrix"
         (with-release [a (ge factory 3 2 (range 6))
                        b (ge factory 3 2 [0 3 1 4 2 5] {:layout :row})]
           (axpy! -1 a b) => (ge factory 3 2)
           (axpy! 2.0 (ge factory 3 2 (range 0 60 10)) a) => (ge factory 3 2 (range 0 121 21))
           (identical? (axpy! 3.0 (ge factory 3 2 (range 6)) a) a) => true
           (axpy! 2.0 (ge factory 2 3 (range 1 7)) a) => (throws ExceptionInfo)
           (axpy! (ge factory 1 3 [10 20 30]) (ge factory 1 3 [1 2 3])) => (ge factory 1 3 [11 22 33]))

         (axpy! 2 (ge factory 2 3 (range 6)) (ge factory 2 3 (range 6))
                (ge factory 2 3 (range 2 9)) 4.0 (ge factory 2 3 (range 8 14)))
         => (ge factory 2 3 (range 34 75 8))

         (axpy! 2 (ge factory 2 3 (range 6)) (ge factory 2 3 (range 6))
                (ge factory 2 3 (range 2 5)) 4.0)
         => (throws ExceptionInfo)

         (axpy! (ge factory 2 3 (range 6)) (ge factory 2 3 (range 6))
                (ge factory 2 3 (range 6)) (ge factory 2 3 (range 6))
                (ge factory 2 3 (range 6)))
         => (axpy! 5 (ge factory 2 3 (range 6)) (ge factory 2 3))

         (axpy! 4 "af" (ge factory 2 3 (range 6)) 3 "b" "c") => (throws ExceptionInfo)

         (with-release [a (ge factory 2 3 (range 6))]
           (axpy! 2 (ge factory 2 3 (range 6)) a (ge factory 2 3 (range 1 7))
                  4.0 (ge factory 2 3 (range 6))) => a)

         (with-release [a (ge factory 3 2 (range 6))]
           (axpy 2.0 (ge factory 2 3 (range 10 70 10))) => (throws ExceptionInfo)
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
              (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4])) => (throws ExceptionInfo)

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

         (mv! (ge factory 2 3 [1 3 5 2 4 6] {:layout :row}) (vctr factory 1 2 3) (vctr factory 0 0))
         => (mv! (ge factory 2 3 [1 2 3 4 5 6] {:layout :column}) (vctr factory 1 2 3) (vctr factory 0 0))

         (mv 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:layout :row}) (vctr factory 1 2 3) (vctr factory 0 0))
         => (vctr factory 44 56)))

(defn test-rk [factory]
  (facts "BLAS 2 rk!"
         (with-release [a (ge factory 2 3)
                        b (ge factory 2 3 {:layout :row})]
           (identical? (rk! 2.0 (vctr factory 1 2) (vctr factory 1 2 3) a) a)
           (identical? (rk! 2.0 (vctr factory 1 2) (vctr factory 1 2 3) b) b))
         => true

         (rk! 1.0 (vctr factory 3 2 1 4) (vctr factory 1 2 3)
              (ge factory 4 3 [1 2 3 4 2 2 2 2 3 4 2 1]))
         => (ge factory 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

         (rk! 1.0 (vctr factory 1 2) (vctr factory 1 2 3) (ge factory 2 2 [1 2 3 5]))
         => (throws ExceptionInfo)))

;; ====================== BLAS 3 ================================

(defn test-ge-mm [factory]
  (facts "BLAS 3 GE mm!"
         (mm! 2.0 (ge factory 3 2 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws ExceptionInfo)

         (mm! 2.0 (ge factory 3 2 [1 2 3 4 5 6]) (ge factory 2 3 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws ExceptionInfo)

         (with-release [c (ge factory 2 2 [1 2 3 4])]
           (identical? (mm! 2.0 (ge factory 2 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6]) 3.0 c) c))
         => true

         (mm! 2.0 (ge factory 2 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [47 62 107 140])

         (mm! 2.0 (ge factory 3 5 (take 15 (repeat 1))) (ge factory 5 3 (take 15 (repeat 1)))
              3.0 (ge factory 3 3 (take 9 (repeat 0)))) => (ge factory 3 3 (take 9 (repeat 10)))

         (mm! 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:layout :row}) (ge factory 3 2 [1 4 2 5 3 6] {:layout :row})
              3.0  (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [47 62 107 140])

         (mm 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:layout :row}) (ge factory 3 2 [1 4 2 5 3 6] {:layout :row})
             3.0  (ge factory 2 2 [1 2 3 4])) => (throws ClassCastException)

         (mm (ge factory 2 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6]))
         => (ge factory 2 2 (list 22.0 28.0 49.0 64.0))

         (with-release [a (ge factory 2 3 (range 1 7 0.1))
                        b (ge factory 3 2 (range 0.1 9 0.2))
                        c (ge factory 2 3 (range 3 5 0.11))
                        d (ge factory 3 2 (range 2 19))
                        ab (mm a b)
                        cd (mm c d)
                        abcd (mm ab cd)
                        abcd-comp (mm a b c d)]
           abcd-comp => abcd)

         (with-release [a (ge factory 2 2 (range 1 7 0.1))
                        b (ge factory 2 2 (range 0.1 9 0.2))
                        c (ge factory 2 2 (range 3 5 0.11))
                        d (ge factory 2 2 (range 2 19))
                        ab (mm a b)
                        abc (mm ab c)
                        abcd (mm abc d)
                        abcd-comp (mm a b c d)]
           abcd-comp => abcd)))

;; ====================== TR Matrix ============================

(defn test-tr-constructor [factory]
  (facts "Create a triangular matrix."
         (with-release [a (tr factory 3 (range 6))
                        b (ge factory 3 3 [0 1 2 0 3 4 0 0 5])]
           (view a) => a
           (view-tr b) => a
           (view-ge (view-tr b)) => b
           (tr factory 3 nil) => (zero a))))

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
           (col a-lower 2) => (vctr factory [5])
           (dia a-upper) => (vctr factory 0 2 5)
           (trans (trans a-upper)) => a-upper
           (trans (trans a-lower)) => a-lower)))

(defn test-tr-entry [factory tr]
  (facts "Triangular Matrix entry."
         (with-release [a-upper (tr factory 3 (range 6) {:uplo :upper})
                        b-diag (tr factory 3 (range 6) {:diag :unit})]
           (entry a-upper 0 1) => 1.0
           (entry a-upper 1 0) => 0.0
           (entry a-upper 1 1) => 2.0
           (entry a-upper 2 0) => 0.0
           (entry a-upper 0 2) => 3.0
           (entry a-upper -1 2) => (throws ExceptionInfo)
           (entry a-upper 3 2) => (throws ExceptionInfo)
           (entry b-diag 2 2) => 1.0
           (entry b-diag 2 1) => 2.0)))

(defn test-tr-entry! [factory tr]
  (facts "Triangular matrix entry!."
         (with-release [a (tr factory 3 [1 2 3 4 5 6] {:diag :unit :uplo :upper})
                        c (ge factory 3 3 (range 1 10))]
           (entry (entry! a 0 1 88.0) 0 1) => 88.0
           (entry (entry! a 1 0 3.0) 0 1) => (throws ExceptionInfo)
           (entry (entry! a 1 0 4.0) 1 1) => (throws ExceptionInfo)
           (entry (view-ge (entry! (view-tr c {:diag :unit}) -1)) 2 2) => 9.0)))

(defn test-tr-bulk-entry! [factory tr]
  (facts "Bulk Triangular matrix entry!."
         (sum (entry! (tr factory 3 [1 2 3 4 5 6]) 33.0)) => 198.0
         (sum (entry! (tr factory 3 [1 2 3 4 5 6] {:diag :unit :uplo :upper}) 22.0)) => 69.0))

(defn test-tr-alter! [factory tr]
  (facts "Triangular alter!."
         (entry (alter! (tr factory 3 [1 2 3 4 5 6] {:layout :row}) 1 1 val+) 1 1) => 4.0
         (alter! (tr factory 3 [1 2 3 4 5 6]) val-ind+) => (tr factory 3 [0 0 0 5 7 16])
         (alter! (tr factory 3 [1 2 3 4 5 6] {:layout :row}) val-ind+)
         => (tr factory 3 [0 0 4 0 7 16] {:layout :row})))

(defn test-tr-dot [factory tr]
  (facts "BLAS 1 Triangular dot"
         (with-release [a (tr factory 3 (range -3 3))
                        b (tr factory 3 [1 2 3 4 5 6])
                        d (tr factory 3 {:layout :row})
                        zero-point (tr factory 0 [])]
           (sqrt (dot a a)) => (roughly (nrm2 a))
           (dot a b) => 7.0
           (dot a (copy! b d)) => (dot a b)
           (dot zero-point zero-point) => 0.0)))

(defn test-tr-nrm2 [factory tr]
  (facts "BLAS 1 Triangular nrm2."
         (nrm2 (tr factory 3 (range -5 5))) => (roughly (sqrt 55))
         (nrm2 (tr factory 0 [])) => 0.0))

(defn test-tr-amax [factory tr]
  (facts "BLAS 1 Triangular amax."
         (amax (tr factory 3 [1 2 3 -4 -3 1 1 1 1 1 1 1])) => 4.0
         (amax (tr factory 0 [])) => 0.0))

(defn test-tr-asum [factory tr]
  (facts "BLAS 1 Triangular asum."
         (asum (tr factory 3 (range -3 3))) => 9.0
         (asum (tr factory 0 [])) => 0.0))

(defn test-tr-sum [factory tr]
  (facts "BLAS 1 Triangular sum."
         (sum (tr factory 3 (range -3 3))) => -3.0
         (sum (tr factory 0 [])) => 0.0))

(defn test-uplo-copy [factory uplo]
  (facts "BLAS 1 copy! uplo matrix"
         (with-release [a (uplo factory 3)
                        b (uplo factory 3 (range 6) {:layout :column})
                        b-row (uplo factory 3 (range 6) {:layout :row :uplo :upper})]

           (identical? (copy a) a) => false
           (identical? (copy! (uplo factory 3 [1 2 3 4 5 6]) a) a) => true
           (copy (uplo factory 3 [1 2 3 4 5 6])) => a
           (copy! b b-row) => (if (.isSymmetric ^MatrixImplementation b) b (throws ExceptionInfo)))

         (copy! (uplo factory 3 [10 20 30 40 50 60]) (uplo factory 3 [1 2 3 4 5 6]))
         => (uplo factory 3 [10 20 30 40 50 60])

         (copy! (uplo factory 3 [1 2 3 4 5 6]) nil) => (throws ExceptionInfo)
         (copy! (uplo factory 3 [10 20 30 40 50 60]) (uplo factory 2)) => (throws ExceptionInfo)))

(defn test-uplo-swap [factory uplo]
  (facts
   "BLAS 1 swap! uplo matrix" uplo-name
   (with-release [a (uplo factory 3 [1 2 3 4 5 6])]
     (identical? (swp! a (uplo factory 3)) a) => true
     (swp! a (uplo factory 3 [10 20 30 40 50 60])) => (uplo factory 3 [10 20 30 40 50 60])
     (swp! a nil) => (throws ExceptionInfo)
     (identical? (swp! a (uplo factory 3 [10 20 30 40 50 60])) a) => true
     (swp! a (uplo factory 2 [10 20])) => (throws ExceptionInfo))))

(defn test-uplo-scal [factory uplo]
  (facts "BLAS 1 scal! uplo matrix"
         (with-release [a (uplo factory 3 [1 2 3 4 5 6])]
           (identical? (scal! 3 a) a) => true)
         (scal! 3 (uplo factory 3 [1 -2 3 9 8 7])) => (uplo factory 3 [3 -6 9 27 24 21])))

(defn test-uplo-axpy [factory uplo]
  (facts "BLAS 1 axpy! uplo matrix"
         (with-release [a (uplo factory 3 (range 6))]

           (axpy! 2.0 (uplo factory 3 (range 0 60 10)) a) => (uplo factory 3 (range 0 121 21))
           (identical? (axpy! 3.0 (uplo factory 3 (range 6)) a) a) => true
           (axpy! 2.0 (uplo factory 2 (range 1 7)) a) => (throws ExceptionInfo)

           (axpy! (uplo factory 2 [10 20 30]) (uplo factory 2 [1 2 3])) => (uplo factory 2 [11 22 33]))

         (axpy! 2 (uplo factory 3 (range 6)) (uplo factory 3 (range 6))
                (uplo factory 3 (range 2 9)) 4.0 (uplo factory 3 (range 8 14)))
         => (uplo factory 3 (range 34 75 8))

         (axpy! 2 (uplo factory 3 (range 6)) (uplo factory 3 (range 6))
                (uplo factory 2 3 (range 2 5)) 4.0) => (throws ExceptionInfo)

         (axpy! (uplo factory 3 (range 6)) (uplo factory 3 (range 6))
                (uplo factory 3 (range 6)) (uplo factory 3 (range 6))
                (uplo factory 3 (range 6)))
         => (axpy! 5 (uplo factory 3 (range 6)) (uplo factory 3))

         (axpy! 4 "af" (uplo factory 3 (range 6)) 3 "b" "c") => (throws ExceptionInfo)

         (with-release [a (uplo factory 3 (range 6))]
           (axpy! 2 (uplo factory 3 (range 6)) a
                  (uplo factory 3 (range 1 7)) 4.0 (uplo factory 3 (range 6))) => a)

         (with-release [a (uplo factory 3 (range 6))]
           (axpy 2.0 (uplo factory 3 (range 10 70 10))) => (throws ExceptionInfo)
           (identical? (axpy 3.0 (uplo factory 3 (range 1 7)) a) a) => false

           (axpy (uplo factory 3 (range 1 7)) (uplo factory 3 (range 2 8))
                 (uplo factory 3 (range 3 9)) (uplo factory 3))
           => (uplo factory 3 (range 6 25 3))

           (axpy 2 (uplo factory 3 (range 1 7)) (uplo factory 3 (range 2 8))) => (uplo factory 3 (range 4 23 3)))))

(defn test-tr-mv [factory tr]
  (facts "BLAS 2 Triangular mv!"
         (mv! 2.0 (tr factory 2 [1 2 3 4])
              (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4])) => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3])]
           (identical? (mv! (tr factory 3 [1 2 3 4 5 6]) y) y)) => true

         (mv! (tr factory 3 [1 10 2 20 3 30] {:layout :row}) (vctr factory 7 0 4))
         => (vctr factory 7 70 260)

         (mv! (tr factory 3 [1 2 3 4 5 6] {:diag :unit :uplo :upper})
              (vctr factory 1 2 3)) => (vctr factory 9 11 3)

         (mv! (tr factory 3 [1 2 3 4 5 6]) (vctr factory 1 2 3))
         => (mv (tr factory 3 [1 2 3 4 5 6]) (vctr factory 1 2 3))

         (mv! (tr factory 3 [1 2 4 3 5 6] {:layout :row}) (vctr factory 1 2 3))
         => (mv! (tr factory 3 [1 2 3 4 5 6] {:layout :column}) (vctr factory 1 2 3))))

(defn test-tr-mm [factory tr]
  (facts "BLAS 3 TR mm!"
         (mm! 2.0 (tr factory 3 [1 2 3 4 5 6]) (tr factory 3 [1 2 3 4 5 6])
              3.0 (tr factory 3 [1 2 3 4 5 6]))=> (throws ExceptionInfo)

         (mm! 2.0 (tr factory 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws ExceptionInfo)

         (with-release [c (ge factory 2 2 [1 2 3 4])]
           (identical? (mm! 2.0 (tr factory 2 [1 2 3 4]) c) c)) => true

         (mm! 2.0 (ge factory 2 3 (range 6)) (tr factory 3 (range 6)))
         => (mm! 2.0 (ge factory 2 3 (range 6)) (ge factory 3 3 [0 1 2 0 3 4 0 0 5]) 0.0 (ge factory 2 3))

         (mm (tr factory 3 (range 6)) (ge factory 3 2 (range 6)))
         => (ge factory 3 2 [0 3 14 0 15 47])

         (mm (ge factory 2 3 (range 6)) (tr factory 3 (range 6)))
         => (ge factory 2 3 [10 13 22 29 20 25])

         (mm 2.0 (tr factory 3 [1 2 3 4 5 6]) (tr factory 3 [1 2 3 4 5 6])
             3.0 (tr factory 3 [1 2 3 4 5 6])) => (throws ClassCastException)))

;; ====================== SY Matrix ============================

(defn test-sy-constructor [factory]
  (facts "Create a SY matrix."
         (with-release [a (sy factory 3 [1 2 3 4 5 6])
                        b (ge factory 3 3 [1 2 3 2 4 5 3 5 6])]
           (view a) => a
           (view-sy b) => a
           (view-ge (view-sy b)) => b
           (sy factory 3 nil) => (zero a)
           (trans (trans a)) => a)))

(defn test-sy [factory]
  (facts "SY Matrix methods."
         (with-release [a-upper (sy factory 3 (range 15) {:uplo :upper})
                        a-lower (sy factory 3 (range 15) {:uplo :lower})]
           (= 3 (mrows a-upper) (ncols a-lower)) => true
           (row a-upper 0) => (vctr factory [0 1 3])
           (row a-upper 2) => (vctr factory [5])
           (col a-upper 0) => (vctr factory [0])
           (col a-upper 2) => (vctr factory [3 4 5])
           (row a-lower 0) => (vctr factory [0])
           (row a-lower 2) => (vctr factory [2 4 5])
           (col a-lower 0) => (vctr factory [0 1 2])
           (col a-lower 2) => (vctr factory [5])
           (dia a-upper) => (vctr factory 0 2 5)
           (trans (trans a-upper)) => a-upper
           (trans (trans a-lower)) => a-lower)))

(defn test-sy-mv [factory mv]
  (facts "BLAS 2 SY mv!"
         (mv! 2.0 (sy factory 2 [1 2 3 4]) (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4]))
         => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3])]
           (identical? (mv! 2 (sy factory 3 [1 2 3 4 5 6]) (vctr factory 1 2 3) 3 y) y)) => true

         (mv! 3 (sy factory 3 [1 10 2 20 3 30] {:layout :row}) (vctr factory 7 0 4) 3 (vctr factory 3))
         => (vctr factory 261 246 780)

         (mv! 2 (sy factory 3 [1 2 3 4 5 6] {:uplo :upper}) (vctr factory 1 2 3) 3 (vctr factory 3))
         => (vctr factory 34 46 64)

         (mv! (sy factory 3 [1 2 4 3 5 6] {:layout :row}) (vctr factory 1 2 3) (vctr factory 3))
         => (mv! (sy factory 3 [1 2 3 4 5 6] {:layout :column}) (vctr factory 1 2 3) (vctr factory 3))))

(defn test-sy-rk [factory sy]
  (facts "BLAS 2 rk!"
         (rk! 2.0 (vctr factory 1 3 4) (sy factory 3 [10 20 30 40 50 60]))
         => (transfer! (rk! 2.0 (vctr factory 1 3 4) (vctr factory 1 3 4)
                            (ge factory 3 3 [10 20 30 20 40 50 30 30 60]))
                       (sy factory 3))

         (rk! 1.0 (vctr factory 1 2) (sy factory 3)) => (throws ExceptionInfo)))

(defn test-sy-mm [factory]
  (facts "BLAS 3 SY mm!"
         (mm! 2.0 (sy factory 3 [1 2 3 4 5 6]) (ge factory 3 3 (range 1 10))
              3.0 (ge factory 3 3 (range 1 10))) => (ge factory 3 3 [31 56 71 76 131 164 121 206 257])

         (mm! 2.0 (sy factory 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (ge factory 3 2 [31 56 71 76 131 164])

         (with-release [c (ge factory 2 2 [1 2 3 4])]
           (identical? (mm! 2.0 (sy factory 2 [1 2 3 4]) c 0.0 c) c)) => true

         (mm! 2.0 (ge factory 2 3 (range 6)) (sy factory 3 (range 6)) 0.0 (ge factory 2 3))
         => (mm! 2.0 (ge factory 2 3 (range 6)) (ge factory 3 3 [0 1 2 1 3 4 2 4 5]) 0.0 (ge factory 2 3))

         (mm (sy factory 3 (range 6)) (ge factory 3 2 (range 6)))
         => (ge factory 3 2 [5 11 14 14 35 47])

         (mm (ge factory 2 3 (range 6)) (sy factory 3 (range 6)))
         => (ge factory 2 3 [10 13 22 30 28 39])

         (mm 2.0 (sy factory 3 [1 2 3 4 5 6]) (sy factory 3 [1 2 3 4 5 6])
             3.0 (sy factory 3 [1 2 3 4 5 6])) => (throws ClassCastException)))

(defn test-sy-mmt [factory sy]
  (facts "BLAS 3 SY rk (mmt!)"
         (with-release [a1 (ge factory 3 7 (range))
                        s1 (sy factory 3 (range))
                        s2 (sy factory 3 (range))]
           (mmt! 2.0 a1 3.0 s1) => (view-sy (mm! 2.0 a1 (trans a1) 3.0 (view-ge s2))))

         (with-release [a1 (ge factory 3 7 (range))
                        s1 (sy factory 3 (range))
                        s2 (sy factory 3 (range))]
           (mmt! 2.0 a1 3.0 (trans s1)) => (view-sy (trans (mm! 2.0 a1 (trans a1) 3.0 (view-ge (trans s2))))))

         (with-release [a1 (ge factory 3 7 (range))
                        s3 (sy factory 7 (range))
                        s4 (sy factory 7 (range))]
           (mmt! 2.0 (trans a1) 3.0 s3) => (view-sy (mm! 2.0 (trans a1) a1 3.0 (view-ge s4))))

         (mmt! 2.0 (sy factory 3 [1 2 3 4 5 6]) 3.0 (sy factory 3 [1 2 3 4 5 6]))
         => (throws ClassCastException)))

(defn test-sy-entry [factory sy]
  (facts "Symmetric Matrix entry."
         (with-release [a-upper (sy factory 3 (range 6) {:uplo :upper})]
           (entry a-upper 0 1) => 1.0
           (entry a-upper 1 0) => 1.0
           (entry a-upper 1 1) => 2.0
           (entry a-upper 2 0) => 3.0
           (entry a-upper 0 2) => 3.0
           (entry a-upper -1 2) => (throws ExceptionInfo)
           (entry a-upper 3 2) => (throws ExceptionInfo))))

(defn test-sy-entry! [factory sy]
  (facts "Symmetric matrix entry!."
         (with-release [a (sy factory 3 [1 2 3 4 5 6] {:uplo :upper})
                        c (ge factory 3 3 (range 1 10))]

           (entry (view-ge (entry! (view-sy c) -1)) 1 2) => 8.0
           (entry (entry! a 0 1 88.0) 0 1) => 88.0
           (entry (entry! a 1 0 3.0) 0 1) => (throws ExceptionInfo)
           (entry (entry! a 1 0 4.0) 1 1) => (throws ExceptionInfo))))

(defn test-sy-bulk-entry! [factory sy]
  (facts "Bulk Symmetric matrix entry!."
         (sum (entry! (sy factory 3 [1 2 3 4 5 6]) 33.0)) => 297.0
         (sum (entry! (sy factory 3 [1 2 3 4 5 6] {:uplo :upper}) 22.0)) => 198.0))

(defn test-sy-alter! [factory sy]
  (facts "Symmetric alter!."
         (entry (alter! (sy factory 3 [1 2 3 4 5 6] {:layout :row}) 1 1 val+) 1 1) => 4.0
         (alter! (sy factory 3 [1 2 3 4 5 6]) val-ind+) => (sy factory 3 [0 0 0 5 7 16])
         (alter! (sy factory 3 [1 2 3 4 5 6] {:layout :row}) val-ind+)
         => (sy factory 3 [0 0 4 0 7 16] {:layout :row})))

(defn test-sy-dot [factory]
  (facts "BLAS 1 SY dot"
         (with-release [a (sy factory 3 (range -3 3))
                        b (sy factory 3 [1 2 3 4 5 6])
                        d (sy factory 3 {:layout :row})
                        zero-point (sy factory 0 [])]
           (sqrt (dot a a)) => (roughly (nrm2 a))
           (dot a b) => 5.0
           (dot a (copy! b d)) => (dot a b)
           (dot zero-point zero-point) => 0.0)))

(defn test-sy-nrm2 [factory sy]
  (facts "BLAS 1 SY nrm2."
         (nrm2 (sy factory 3 (range -5 5))) => (roughly (sqrt 81))
         (nrm2 (sy factory 0 [])) => 0.0))

(defn test-sy-amax [factory sy]
  (facts "BLAS 1 SY amax."
         (amax (sy factory 3 [1 2 3 -4 -3 1 1 1 1 1 1 1])) => 4.0
         (amax (sy factory 0 [])) => 0.0))

(defn test-sy-asum [factory sy]
  (facts "BLAS 1 SY asum."
         (asum (sy factory 3 (range -3 3))) => 13.0
         (asum (sy factory 0 [])) => 0.0))

(defn test-sy-sum [factory sy]
  (facts "BLAS 1 SY sum."
         (sum (sy factory 3 (range -3 3))) => -5.0
         (sum (sy factory 0 [])) => 0.0))

;; ==================== Banded Matrix ======================================

(defn test-gb-constructor [factory]
  (facts "Create a GB matrix."
         (with-release [a (gb factory 5 8 2 1 (range 1 100))
                        a-ge (ge factory 4 6 [0.00    4.00    8.00   12.00   15.00   17.00
                                              1.00    5.00    9.00   13.00   16.00    0.00
                                              2.00    6.00   10.00   14.00    0.00    0.00
                                              3.00    7.00   11.00    0.00    0.00    0.00]
                                 {:layout :row})
                        b (gb factory 5 8 4 7 (range 1 100))
                        b-ge (ge factory 12 8 [0.00    0.00    0.00    0.00    0.00    0.00    0.00   36.00
                                               0.00    0.00    0.00    0.00    0.00    0.00   31.00   37.00
                                               0.00    0.00    0.00    0.00    0.00   26.00   32.00   38.00
                                               0.00    0.00    0.00    0.00   21.00   27.00   33.00   39.00
                                               0.00    0.00    0.00   16.00   22.00   28.00   34.00   40.00
                                               0.00    0.00   11.00   17.00   23.00   29.00   35.00    0.00
                                               0.00    6.00   12.00   18.00   24.00   30.00    0.00    0.00
                                               1.00    7.00   13.00   19.00   25.00    0.00    0.00    0.00
                                               2.00    8.00   14.00   20.00    0.00    0.00    0.00    0.00
                                               3.00    9.00   15.00    0.00    0.00    0.00    0.00    0.00
                                               4.00   10.00    0.00    0.00    0.00    0.00    0.00    0.00
                                               5.00    0.00    0.00    0.00    0.00    0.00    0.00    0.00]
                                 {:layout :row})
                        c (gb factory 5 8 4 1 (range 1 100))
                        c-ge (ge factory 6 6 [0.00    6.00   11.00   15.00   18.00   20.00
                                              1.00    7.00   12.00   16.00   19.00    0.00
                                              2.00    8.00   13.00   17.00    0.00    0.00
                                              3.00    9.00   14.00    0.00    0.00    0.00
                                              4.00   10.00    0.00    0.00    0.00    0.00
                                              5.00    0.00    0.00    0.00    0.00    0.00]
                                 {:layout :row})
                        d (gb factory 5 2 3 1 (range 1 100))
                        d-ge (ge factory 5 2 [0.00    5.00
                                              1.00    6.00
                                              2.00    7.00
                                              3.00    8.00
                                              4.00    9.00 ]
                                 {:layout :row})
                        a1 (gb factory 5 8 2 1 (range 1 100) {:layout :row})
                        a1-ge (ge factory 5 4 [0.00    0.00    1.00    2.00
                                               0.00    3.00    4.00    5.00
                                               6.00    7.00    8.00    9.00
                                               10.00   11.00   12.00   13.00
                                               14.00   15.00   16.00   17.00]
                                  {:layout :row})
                        b1 (gb factory 5 8 4 7 (range 1 100) {:layout :row})
                        b1-ge (ge factory 5 12 [0.00 0.00 0.00 0.00 1.00 2.00 3.00 4.00 5.00 6.00 7.00 8.00
                                                0.00 0.00 0.00 9.00 10.00 11.00 12.00 13.00 14.00 15.00 16.00 0.00
                                                0.00 0.00 17.00 18.00 19.00 20.00 21.00 22.00 23.00 24.00 0.00 0.00
                                                0.00 25.00 26.00 27.00 28.00 29.00 30.00 31.00 32.00 0.00 0.00 0.00
                                                33.00 34.00 35.00 36.00 37.00 38.00 39.00 40.00 0.00 0.00 0.00 0.00]
                                  {:layout :row})
                        c1 (gb factory 5 8 1 4 (range 1 100) {:layout :row})
                        c1-ge (ge factory 5 6 [0.00    1.00    2.00    3.00    4.00    5.00
                                               6.00    7.00    8.00    9.00   10.00   11.00
                                               12.00   13.00   14.00   15.00   16.00   17.00
                                               18.00   19.00   20.00   21.00   22.00   23.00
                                               24.00   25.00   26.00   27.00   28.00    0.00]
                                  {:layout :row})
                        d1 (gb factory 5 2 3 1 (range 1 100) {:layout :row})
                        d1-ge (ge factory 5 5 [0.00    0.00    0.00    1.00    2.00
                                               0.00    0.00    3.00    4.00    0.00
                                               0.00    5.00    6.00    0.00    0.00
                                               7.00    8.00    0.00    0.00    0.00
                                               9.00    0.00    0.00    0.00    0.00]
                                  {:layout :row})]
           (view a) => a
           (view-ge a) => a-ge
           (view-ge b) => b-ge
           (view-ge c) => c-ge
           (view-ge d) => d-ge
           (view-ge a1) => a1-ge
           (view-ge b1) => b1-ge
           (view-ge c1) => c1-ge
           (view-ge d1) => d1-ge
           (gb factory 5 8 2 1 nil) => (zero a)
           (gb factory 5 8 5 1) => (throws ExceptionInfo)
           (gb factory 5 8 2 8) => (throws ExceptionInfo))))

(defn test-gb [factory]
  (facts "GB Matrix methods."
         (with-release [a-row (gb factory 4 3 2 1 (range 1 100) {:layout :row})
                        a-col (gb factory 4 3 2 1 (range 1 100))]
           (row a-row 0) => (vctr factory 1 2)
           (row a-row 2) => (vctr factory 6 7 8)
           (col a-row 0) => (vctr factory 1 3 6)
           (col a-row 2) => (vctr factory 5 8 10)
           (dia a-row -2) => (vctr factory 6 9)
           (row a-col 0) => (vctr factory 1 4)
           (row a-col 2) => (vctr factory 3 6 9)
           (col a-col 0) => (vctr factory 1 2 3)
           (col a-col 2) => (vctr factory 8 9 10)
           (dia a-col -2) => (vctr factory 3 7))))

(defn test-gb-copy [factory]
  (facts "BLAS 1 copy! GB matrix"
         (with-release [a (gb factory 5 4 2 3 (range 1 20))
                        b (gb factory 5 4 2 3 (range 100 200))
                        b-row (gb factory 5 4 2 3 {:layout :row})]
           (identical? (copy! a b) b) => true
           (copy! a b) => (gb factory 5 4 2 3 (range 1 20))
           (copy (gb factory 5 4 2 3 (range 1 20))) => a
           (copy! a b-row) => a)

         (copy! (gb factory 5 4 2 3 [1 2 3 4 5 6]) nil) => (throws ExceptionInfo)

         (copy! (gb factory 5 4 2 3 [10 20 30 40 50 60]) (gb factory 5 4 2 2)) => (throws ExceptionInfo)))

(defn test-gb-swap [factory]
  (facts
   "BLAS 1 swap! GB matrix"
   (with-release [a (gb factory 5 4 2 3 (range 1 20))
                  b (gb factory 5 4 2 3 (range 100 200))
                  b-row (gb factory 5 4 2 3 [100 103 107 112
                                             101 104 108 113
                                             102 105 109 114
                                             106 110 115
                                             111 116]
                            {:layout :row})]
     (swp! a (gb factory 5 4 2 3)) => a
     (swp! a (gb factory 5 4 2 3 (range 10 200 10))) => (gb factory 5 4 2 3 (range 10 200 10))
     (swp! a b-row) => b

     (swp! a nil) => (throws ExceptionInfo)
     (identical? (swp! a (gb factory 5 4 2 3 (range 10 200 10))) a) => true
     (swp! a (gb factory 5 4 1 3 (range 10 200 10))) => (throws ExceptionInfo))))

(defn test-gb-scal [factory]
  (facts "BLAS 1 scal! GB matrix"
         (with-release [a (gb factory 5 4 2 3 (range 1 20))]
           (identical? (scal! 3 a) a) => true)
         (scal! 3 (gb factory 5 4 2 3 (range 1 20))) => (gb factory 5 4 2 3 (range 3 60 3))
         (scal! 3 (submatrix (gb factory 5 4 2 3 (range 1 20)) 1 1 2 3))
         => (gb factory 2 3 1 2 [15 18 27 30 42 45])))

(defn test-gb-axpy [factory]
  (facts "BLAS 1 axpy! GB matrix"
         (with-release [a (gb factory 5 6 3 2 (range 1 100))
                        b (gb factory 5 6 3 2
                              [1 5 10 2 6 11 15 3 7 12 16 19 4 8 13 17 20 22 9 14 18 21 23]
                              {:layout :row})]
           (axpy! -1 a b) => (gb factory 5 6 3 2)
           (identical? (axpy! 3.0 (gb factory 5 6 3 2 (range 60)) a) a) => true
           (axpy! 2.0 (gb factory 5 6 2 3 (range 1 70)) a) => (throws ExceptionInfo))))

(defn test-gb-mv [factory]
  (facts "BLAS 2 GB mv!"
         (mv! 2.0 (gb factory 6 5 3 2 (range 1 100))
              (vctr factory 1 2 3 4 5) 3 (vctr factory [1 2 3 4])) => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3 4 5 6])]
           (identical? (mv! 2 (gb factory 6 5 3 2 (range 1 100)) (vctr factory 1 2 3 4 5) 3 y) y))
         => true

         (mv! (gb factory 5 6 2 3 (range 1 100)) (vctr factory 1 2 3 4 5 6) (vctr factory 5))
         => (vctr factory [85.00 185.00 332.00 349.00 353.00])))

(defn test-gb-mm [factory]
  (facts "BLAS 3 GB mm!"
         (mm! 2.0 (gb factory 5 6 3 2 (range 1 100)) (ge factory 6 2 (range 1 13))
              3.0 (ge factory 5 2 (range 10 30)))
         => (ge factory 5 2 [112.00  319.00
                             247.00  670.00
                             460.00 1159.00
                             757.00 1780.00
                             792.00 1827.00]
                {:layout :row})

         (with-release [c (ge factory 5 2)]
           (identical? (mm! 2.0 (gb factory 5 6 2 3 [1 2 3 4 5 6]) (ge factory 6 2 (range 100)) 3.0 c) c))
         => true))

(defn test-gb-entry! [factory]
  (facts "GB matrix entry!."
         (with-release [a (gb factory 5 4 1 2 (range 1 100))
                        c (ge factory 4 4 [0.00    0.00    6.00   10.00
                                           0.00    3.00    7.00   11.00
                                           1.00    4.00    8.00   12.00
                                           2.00    5.00    9.00   13.00 ]
                              {:layout :row})]
           (view-ge (entry! (view-sy c) -1)) => c
           (entry c 1 2) => 7.0
           (entry (entry! a 2 1 88.0) 2 1) => 88.0
           (entry! a 0 4 3.0) => (throws ExceptionInfo)
           (entry! a 4 0 4.0) => (throws ExceptionInfo))))

(defn test-gb-alter! [factory]
  (facts "GB alter!."
         (entry (alter! (gb factory 5 3 1 2 (range 100) {:layout :row}) 1 1 val+) 1 1) => 5.0
         (alter! (gb factory 4 4 0 3 (range 100)) val-ind+) => (gb factory 4 4 0 3 [0 1 3 6 10 14 18 24 30 36])
         (alter! (gb factory 100 2 2 1 (range 1 300) {:layout :row}) val-ind+)
         => (gb factory 100 2 2 1 [0 2 0 5 0 8 10] {:layout :row})))

(defn test-gb-dot [factory]
  (facts "BLAS 1 GB dot"
         (with-release [a (gb factory 5 2 1 1 (range -10 100))
                        b (gb factory 5 2 1 1 (range 1 100))
                        d (gb factory 5 2 1 1 {:layout :row})
                        zero-point (gb factory 5 2 1 1)]
           (sqrt (dot a a)) => (roughly (nrm2 a) 0.000001)
           (dot a b) => -110.0
           (dot a (copy! b d)) => (dot a b)
           (dot zero-point zero-point) => 0.0)))

(defn test-gb-nrm2 [factory]
  (facts "BLAS 1 GB nrm2."
         (nrm2 (gb factory 5 4 2 3 (repeat 1.0))) => (roughly (sqrt 17))
         (nrm2 (gb factory 1 1 0 0 [])) => 0.0))

(defn test-gb-amax [factory]
  (facts "BLAS 1 GB amax."
         (amax (gb factory 5 5 2 3 (range 1 100))) => 21.0
         (amax (gb factory 0 0 0 0 [])) => 0.0))

(defn test-gb-asum [factory]
  (facts "BLAS 1 GB asum."
         (asum (gb factory 5 4 2 3 (range 1 100))) => 153.0
         (asum (gb factory 0 0 0 0 [])) => 0.0))

(defn test-gb-sum [factory]
  (facts "BLAS 1 gb sum."
         (sum (gb factory 5 4 2 3 (range 1 100))) => 153.0
         (sum (gb factory 0 0 0 0 [])) => 0.0))

(defn test-banded-uplo-constructor [factory uplo]
  (facts "Create a banded uplo matrix."
         (with-release [a (uplo factory 4 2 [1 2 3 4 5 6 7 8 9])
                        b (ge factory 3 4 [1 2 3 4 5 6 7 8 0 9 0 0])]
           (view a) => a
           (view-ge a) => b
           (uplo factory 4 2 nil) => (zero a))))

(defn test-banded-uplo [factory uplo]
  (facts "Banded uplo Matrix methods."
         (with-release [a-upper (uplo factory 4 2 (range 15) {:layout :row :uplo :upper})
                        a-lower (uplo factory 4 2 (range 15) {:uplo :lower})]
           (= 4 (mrows a-upper) (ncols a-lower)) => true
           (row a-upper 0) => (vctr factory [0 1 2])
           (row a-upper 2) => (vctr factory [6 7])
           (col a-upper 0) => (vctr factory [0])
           (col a-upper 2) => (vctr factory [2 4 6])
           (row a-lower 0) => (vctr factory [0])
           (row a-lower 2) => (vctr factory [2 4 6])
           (col a-lower 0) => (vctr factory [0 1 2])
           (col a-lower 2) => (vctr factory [6 7])
           (dia a-upper) => (vctr factory 0 3 6 8)
           (uplo factory 4 2 (range 15) {:uplo :upper}) => (throws ExceptionInfo)
           (trans (trans a-upper)) => a-upper
           (trans (trans a-lower)) => a-lower)))

(defn test-sb-entry [factory]
  (facts "SB Matrix entry."
         (with-release [a-upper (sb factory 3 (range 6) {:uplo :upper :layout :row})]
           (entry a-upper 0 1) => 1.0
           (entry a-upper 1 0) => 1.0
           (entry a-upper 1 1) => 3.0
           (entry a-upper 2 0) => 2.0
           (entry a-upper 0 2) => 2.0
           (entry a-upper -1 2) => (throws ExceptionInfo)
           (entry a-upper 3 2) => (throws ExceptionInfo))))

(defn test-sb-entry! [factory]
  (facts "SB matrix entry!."
         (with-release [a (sb factory 3 [1 2 3 4 5 6] {:uplo :upper :layout :row})
                        c (ge factory 3 3 (range 1 10))]

           (entry (view-ge (entry! (view-sy c) -1)) 1 2) => 8.0
           (entry (entry! a 0 1 88.0) 0 1) => 88.0
           (entry (entry! a 1 0 3.0) 0 1) => (throws ExceptionInfo)
           (entry (entry! a 1 0 4.0) 1 1) => (throws ExceptionInfo))))

(defn test-sb-bulk-entry! [factory]
  (facts "Bulk SB matrix entry!."
         (sum (entry! (sb factory 3 [1 2 3 4 5 6]) 33.0)) => 297.0
         (sum (entry! (sb factory 3 [1 2 3 4 5 6] {:layout :row :uplo :upper}) 22.0)) => 198.0))

(defn test-sb-alter! [factory]
  (facts "SB alter!."
         (entry (alter! (sb factory 3 [1 2 3 4 5 6] {:layout :row :uplo :upper}) 1 1 val+) 1 1) => 5.0
         (alter! (sb factory 3 [1 2 3 4 5 6]) val-ind+) => (sb factory 3 [0 0 0 5 7 16])
         (alter! (sb factory 3 [1 2 3 4 5 6] {:layout :row :uplo :upper}) val-ind+)
         => (sb factory 3 [0 2 6 5 12 16] {:layout :row :uplo :upper})))

(defn test-tb-entry [factory]
  (facts "TB Matrix entry."
         (with-release [a (tb factory 3 (range 6) {:uplo :upper :layout :row})]
           (entry a 0 1) => 1.0
           (entry a 1 0) => 0.0
           (entry a 1 1) => 3.0
           (entry a 2 0) => 0.0
           (entry a 0 2) => 2.0
           (entry a -1 2) => (throws ExceptionInfo)
           (entry a 3 2) => (throws ExceptionInfo))))

(defn test-tb-entry! [factory]
  (facts "TB matrix entry!."
         (with-release [a (tb factory 3 [1 2 3 4 5 6] {:uplo :upper :layout :row})
                        c (ge factory 3 3 (range 1 10))]

           (entry (view-ge (entry! (view-tr c) -1)) 1 2) => 8.0
           (entry (entry! a 0 1 88.0) 0 1) => 88.0
           (entry (entry! a 1 0 3.0) 0 1) => (throws ExceptionInfo)
           (entry (entry! a 1 0 4.0) 1 1) => (throws ExceptionInfo))))

(defn test-tb-bulk-entry! [factory]
  (facts "Bulk TB matrix entry!."
         (sum (entry! (tb factory 3 [1 2 3 4 5 6]) 33.0)) => 198.0
         (sum (entry! (tb factory 3 [1 2 3 4 5 6] {:layout :row :uplo :upper}) 22.0)) => 132.0))

(defn test-tb-alter! [factory]
  (facts "TB alter!."
         (entry (alter! (tb factory 3 [1 2 3 4 5 6] {:layout :row :uplo :upper}) 1 1 val+) 1 1) => 5.0
         (alter! (tb factory 3 [1 2 3 4 5 6]) val-ind+) => (tb factory 3 [0 0 0 5 7 16])
         (alter! (tb factory 3 [1 2 3 4 5 6] {:layout :row :uplo :upper}) val-ind+)
         => (tb factory 3 [0 2 6 5 12 16] {:layout :row :uplo :upper})))

(defn test-tb-mv [factory]
  (facts "BLAS 2 TB mv!"
         (mv! 2.0 (tb factory 2 [1 2 3 4]) (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4]))
         => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3])]
           (identical? (mv! (tb factory 3 [1 2 3 4 5 6]) y) y)) => true

         (mv! (tb factory 3 [1 2 3 10 20 30] {:layout :row :uplo :upper}) (vctr factory 7 0 4))
         => (vctr factory 19 80 120)

         (mv! (tb factory 3 [1 2 3 4 5 6] {:diag :unit :layout :row :uplo :upper})
              (vctr factory 1 2 3))=> (vctr factory 9 11 3)

         (mv! (tb factory 3 [1 2 3 4 5 6]) (vctr factory 1 2 3))
         => (mv (tb factory 3 [1 2 3 4 5 6]) (vctr factory 1 2 3))

         (mv! (tb factory 3 [1 2 4 3 5 6] {:layout :row :uplo :upper}) (vctr factory 1 2 3))
         => (mv! (transfer! (tb factory 3 [1 2 4 3 5 6] {:layout :row :uplo :upper})
                            (tr factory 3 {:layout :row :uplo :upper}))
                 (vctr factory 1 2 3))))

(defn test-tb-mm [factory]
  (facts "BLAS 3 TB mm!"
         (mm! 2.0 (tb factory 3 [1 2 3 4 5 6]) (tb factory 3 [1 2 3 4 5 6])
              3.0 (tb factory 3 [1 2 3 4 5 6]))=> (throws ExceptionInfo)

         (mm! 2.0 (tb factory 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws ExceptionInfo)

         (with-release [c (ge factory 2 2 [1 2 3 4])]
           (identical? (mm! 2.0 (tb factory 2 [1 2 3 4]) c) c))=> true

         (mm! 2.0 (ge factory 2 3 (range 6)) (tb factory 3 (range 6)))
         => (mm! 2.0 (ge factory 2 3 (range 6)) (ge factory 3 3 [0 1 2 0 3 4 0 0 5]) 0.0 (ge factory 2 3))

         (mm (tb factory 3 (range 6)) (ge factory 3 2 (range 6)))
         => (ge factory 3 2 [0 3 14 0 15 47])

         (mm (ge factory 2 3 (range 6)) (tb factory 3 (range 6)))
         => (ge factory 2 3 [10 13 22 29 20 25])

         (mm 2.0 (tb factory 3 [1 2 3 4 5 6]) (tb factory 3 [1 2 3 4 5 6])
             3.0 (tb factory 3 [1 2 3 4 5 6])) => (throws ClassCastException)))

;; ==================== Packed Matrix ======================================

(defn test-packed-copy [factory uplo]
  (facts "BLAS 1 copy! packed matrix"
         (with-release [a (uplo factory 3)
                        b (uplo factory 3 (range 6) {:layout :column})
                        b-row (uplo factory 3 (range 6) {:layout :row})]

           (identical? (copy a) a) => false
           (identical? (copy! (uplo factory 3 [1 2 3 4 5 6]) a) a) => true
           (copy (uplo factory 3 [1 2 3 4 5 6])) => a
           (copy! b b-row) => (throws ExceptionInfo))

         (copy! (uplo factory 3 [10 20 30 40 50 60]) (uplo factory 3 [1 2 3 4 5 6]))
         => (uplo factory 3 [10 20 30 40 50 60])

         (copy! (uplo factory 3 [1 2 3 4 5 6]) nil) => (throws ExceptionInfo)
         (copy! (uplo factory 3 [10 20 30 40 50 60]) (uplo factory 2)) => (throws ExceptionInfo)))

(defn test-packed-constructor [factory pck]
  (facts "Create a packed matrix."
         (with-release [a (pck factory 3 (range 6))]
           (entry a 1 1) => 3.0
           (pck factory 3 nil) => (zero a))))

(defn test-packed [factory pck]
  (facts "Packed Matpix methods."
         (with-release [a-upper (tp factory 3 (range 15) {:uplo :upper})
                        b-upper (tp factory 3 (range 15) {:uplo :upper :layout :row})
                        a-lower (tp factory 3 (range 15) {:uplo :lower})
                        b-lower (tp factory 3 (range 15) {:uplo :lower :layout :row})]
           (= 3 (mrows a-upper) (ncols a-lower) (mrows b-upper) (ncols b-lower)) => true
           (row b-upper 0) => (vctr factory [0 1 2])
           (row b-upper 2) => (vctr factory [5])
           (col a-upper 0) => (vctr factory [0])
           (col a-upper 2) => (vctr factory [3 4 5])
           (row b-lower 0) => (vctr factory [0])
           (row b-lower 2) => (vctr factory [3 4 5])
           (col a-lower 0) => (vctr factory [0 1 2])
           (col a-lower 2) => (vctr factory [5])
           (row a-upper 0) => (throws ExceptionInfo)
           (col b-upper 0) => (throws ExceptionInfo)
           (dia a-upper) => (throws ExceptionInfo)
           (trans (trans a-upper)) => a-upper
           (trans (trans a-lower)) => a-lower
           (trans (trans b-upper)) => b-upper
           (trans (trans b-lower)) => b-lower)))

(defn test-tp-dot [factory]
  (facts "BLAS 1 TP dot"
         (with-release [a (tp factory 3 (range -3 3))
                        b (tp factory 3 [1 2 3 4 5 6])
                        zero-point (tp factory 0 [])]
           (sqrt (dot a a)) => (roughly (nrm2 a))
           (dot a b) => 7.0
           (dot zero-point zero-point) => 0.0)))

(defn test-tp-mm [factory]
  (facts "BLAS 3 TP mm!"
         (mm! 2.0 (tp factory 3 [1 2 3 4 5 6]) (tp factory 3 [1 2 3 4 5 6])
              3.0 (tp factory 3 [1 2 3 4 5 6])) => (throws ExceptionInfo)

         (mm! 2.0 (tp factory 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws ExceptionInfo)

         (with-release [c (ge factory 2 2 [1 2 3 4])]
           (identical? (mm! 2.0 (tp factory 2 [1 2 3 4]) c) c)) => true

         (mm! 2.0 (ge factory 2 3 (range 6)) (tp factory 3 (range 6)))
         => (mm! 2.0 (ge factory 2 3 (range 6)) (tr factory 3 (range 6)))

         (mm (tp factory 3 (range 6)) (ge factory 3 2 (range 6)))
         => (ge factory 3 2 [0 3 14 0 15 47])

         (mm (ge factory 2 3 (range 6)) (tp factory 3 (range 6)))
         => (mm (ge factory 2 3 (range 6)) (tr factory 3 (range 6)))

         (mm 2.0 (tp factory 3 [1 2 3 4 5 6]) (tp factory 3 [1 2 3 4 5 6])
             3.0 (tp factory 3 [1 2 3 4 5 6])) => (throws ClassCastException)))

(defn test-sp-dot [factory]
  (facts "BLAS 1 SP dot"
         (with-release [a (sp factory 3 (range -3 3))
                        b (sp factory 3 [1 2 3 4 5 6])
                        zero-point (sp factory 0 [])]
           (sqrt (dot a a)) => (roughly (nrm2 a))
           (dot a b) => 5.0
           (dot zero-point zero-point) => 0.0)))

(defn test-sp-mm [factory]
  (facts "BLAS 3 SP mm!"
         (mm! 2.0 (sp factory 3 [1 2 3 4 5 6]) (ge factory 3 3 (range 1 10))
              3.0 (ge factory 3 3 (range 1 10))) => (ge factory 3 3 [31 56 71 76 131 164 121 206 257])

         (mm! 2.0 (sp factory 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (ge factory 3 2 [31 56 71 76 131 164])

         (with-release [c (ge factory 2 2 [1 2 3 4])]
           (identical? (mm! 2.0 (sp factory 2 [1 2 3 4]) c 0.0 c) c)) => true

         (mm! 2.0 (ge factory 2 3 (range 6)) (sp factory 3 (range 6)) 0.0 (ge factory 2 3))
         => (mm! 2.0 (ge factory 2 3 (range 6)) (sy factory 3 (range 6)) 0.0 (ge factory 2 3))

         (mm (sy factory 3 (range 6)) (ge factory 3 2 (range 6)))
         => (ge factory 3 2 [5 11 14 14 35 47])

         (mm (ge factory 2 3 (range 6)) (sp factory 3 (range 6)))
         => (mm (ge factory 2 3 (range 6)) (sy factory 3 (range 6)))

         (mm 2.0 (sp factory 3 [1 2 3 4 5 6]) (sp factory 3 [1 2 3 4 5 6])
             3.0 (sp factory 3 [1 2 3 4 5 6])) => (throws ClassCastException)))

;; -------------------- Diagonal matrix tests -----------------------------

(defn test-gd-constructor [factory]
  (facts "Create a GD matrix."
         (with-release [a (gd factory 4 [1 2 3 4 5 6 7 8 9])]
           (gd factory 4 nil) => (zero a))))

(defn test-gd [factory]
  (facts "GD Matrix methods."
         (with-release [d (gd factory 4 (range 1 100))
                        b (gb factory 4 4 0 0 (range 1 100))]
           (dia d -2) => (vctr factory 0)
           (dia d) => (dia b)
           (trans (trans d)) => d)))

(defn test-gd-copy [factory]
  (facts "BLAS 1 copy! GD matrix"
         (with-release [a (gd factory 5 (range 1 20))
                        a1 (gd factory 5)
                        b (gb factory 5 5 0 0 (range 100 200))]
           (identical? (copy! a a1) a1) => true
           (identical? (copy! a b) b) => (throws ExceptionInfo)
           (copy! a a1) => (gd factory 5 (range 1 20))
           (copy (gd factory 5 (range 1 20))) => a
           (copy! a nil) => (throws ExceptionInfo)
           (copy! a (gd factory 4)) => (throws ExceptionInfo))))

(defn test-gd-swap [factory]
  (facts
   "BLAS 1 swap! GD matrix"
   (with-release [a (gd factory 5 (range 1 20))]
     (swp! a (gd factory 5)) => a
     (swp! a (gd factory 5 (range 10 200 10))) => (gd factory 5 (range 10 200 10))

     (swp! a nil) => (throws ExceptionInfo)
     (identical? (swp! a (gd factory 5 (range 10 200 10))) a) => true
     (swp! a (gd factory 4 (range 10 200 10))) => (throws ExceptionInfo))))

(defn test-gd-scal [factory]
  (facts "BLAS 1 scal! GD matrix"
         (with-release [a (gd factory 5 (range 1 20))]
           (identical? (scal! 3 a) a) => true)
         (scal! 3 (gd factory 5 (range 1 20))) => (gd factory 5 (range 3 60 3))))

(defn test-gd-axpy [factory]
  (facts "BLAS 1 axpy! GD matrix"
         (with-release [a (gd factory 6 (range 1 100))
                        b (gd factory 6 (range 2 200 2))]
           (axpy -1 a a) => (gd factory 6)
           (identical? (axpy! 3.0 (gd factory 6 (range 60)) a) a) => true
           (axpy! 2.0 (gd factory 5 (range 1 70)) a) => (throws ExceptionInfo))))

(defn test-gd-mv [factory]
  (facts "BLAS 2 GD mv!"
         (mv! 2.0 (gd factory 5 (range 1 100))
              (vctr factory 1 2 3 4 5) 3 (vctr factory [1 2 3 4]))=> (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3 4 5])]
           (identical? (mv! 2 (gd factory 5 (range 1 100)) (vctr factory 1 2 3 4 5) 3 y) y))
         => true

         (with-release [y1 (vctr factory [1 2 3 4 5])
                        y2 (copy y1)
                        d (gd factory 5 (range 1 100))
                        g (let-release [res (ge factory 5 5)] (copy! y1 (dia res)) res)]
           (mv! 2 g (vctr factory 1 2 3 4 5) 3 y1) => (mv! 2 d (vctr factory 1 2 3 4 5) 3 y2)
           (mv! d y1) => (mv g y2))

         (mv! (gd factory 5 (range 1 100)) (vctr factory 1 2 3 4 5) (vctr factory 5))
         => (vctr factory [1 4 9 16 25])))

(defn test-gd-mm [factory]
  (facts "BLAS 3 GD mm!"
         (mm! 2.0 (gd factory 5 (range 1 100)) (ge factory 5 3 (range 1 16)))
         => (mm! 2.0 (tb factory 5 0 (range 1 100)) (ge factory 5 3 (range 1 16)))

         (mm! 2.0 (trans (ge factory 5 3 (range 1 16))) (trans (gd factory 5 (range 1 100))))
         => (trans (mm! 2.0 (tb factory 5 0 (range 1 100)) (ge factory 5 3 (range 1 16))))

         (with-release [c (ge factory 6 2 (range 1 13))]
           (mm! 2.0 (gd factory 6 [1 2 3 4 5 6]) (ge factory 6 2 (range 100)) 3.0 c)
           => (mm! 2.0 (sb factory 6 0 [1 2 3 4 5 6]) (ge factory 6 2 (range 100)) 3.0 c))))

(defn test-gd-entry! [factory]
  (facts "GD matrix entry!."
         (with-release [a (gd factory 5 (range 1 100))]
           (entry a 1 2) => 0.0
           (entry (entry! a 1 1 88.0) 1 1) => 88.0
           (entry! a 0 4 3.0) => (throws ExceptionInfo))))

(defn test-gd-alter! [factory]
  (facts "GD alter!."
         (entry (alter! (gd factory 5 (range 100)) 1 1 val+) 1 1) => 2.0
         (alter! (gd factory 4 (range 100)) val-ind+) => (throws ExceptionInfo)))

(defn test-gd-dot [factory]
  (facts "BLAS 1 GD dot"
         (with-release [a (gd factory 5 (range -10 100))
                        a1 (gb factory 5 5 0 0 (range -10 100))
                        b (gd factory 5 (range 1 100))
                        b1 (gb factory 5 5 0 0 (range 1 100))
                        d (gd factory 5)
                        zero-point (gd factory 5)]
           (sqrt (dot a a)) => (roughly (nrm2 a) 0.000001)
           (dot a b) => (dot a1 b1)
           (dot a (copy! b d)) => (dot a b)
           (dot zero-point zero-point) => 0.0)))

(defn test-gd-nrm2 [factory]
  (facts "BLAS 1 GD nrm2."
         (nrm2 (gd factory 5 (repeat 1.0))) => (nrm2 (gb factory 5 5 0 0 (repeat 1.0)))
         (nrm2 (gd factory 1 [])) => 0.0))

(defn test-gd-amax [factory]
  (facts "BLAS 1 GD amax."
         (amax (gd factory 5 (range 1 100))) => (amax (gb factory 5 5 0 0 (range 1 100)))
         (amax (gd factory 0 [])) => 0.0))

(defn test-gd-asum [factory]
  (facts "BLAS 1 GD asum."
         (asum (gd factory 5 (range 1 100))) => (asum (gb factory 5 5 0 0 (range 1 100)))
         (asum (gd factory 0 [])) => 0.0))

(defn test-gd-sum [factory]
  (facts "BLAS 1 GD sum."
         (sum (gd factory 5 (range 1 100))) => (sum (gb factory 5 5 0 0 (range 1 100)))))

(defn test-gt-constructor [factory gt]
  (facts "Create a GT matrix."
         (with-release [a (gt factory 4 (range 1 100))]
           (gt factory 4 nil) => (zero a))))

(defn test-gt [factory gt]
  (facts "GT Matrix methods."
         (with-release [d (gt factory 4 (range 1 100))]
           (dia d -2) => (vctr factory 0)
           (dia d) => (vctr factory [1 2 3 4])
           (dia d 1) => (vctr factory [5 6 7])
           (dia d -1) => (vctr factory [8 9 10])
           (trans (trans d)) => (throws ExceptionInfo))))

(defn test-gt-copy [factory gt]
  (facts "BLAS 1 copy! GT matrix"
         (with-release [a (gt factory 5 (range 1 20))
                        a1 (gt factory 5)]
           (identical? (copy! a a1) a1) => true
           (copy! a a1) => (gt factory 5 (range 1 20))
           (copy (gt factory 5 (range 1 20))) => a
           (copy! a nil) => (throws ExceptionInfo)
           (copy! a (gt factory 4)) => (throws ExceptionInfo))))

(defn test-gt-swap [factory gt]
  (facts
   "BLAS 1 swap! GT matrix"
   (with-release [a (gt factory 5 (range 1 20))]
     (swp! a (gt factory 5)) => a
     (swp! a (gt factory 5 (range 10 200 10))) => (gt factory 5 (range 10 200 10))

     (swp! a nil) => (throws ExceptionInfo)
     (identical? (swp! a (gt factory 5 (range 10 200 10))) a) => true
     (swp! a (gt factory 4 (range 10 200 10))) => (throws ExceptionInfo))))

(defn test-gt-scal [factory gt]
  (facts "BLAS 1 scal! GT matrix"
         (with-release [a (gt factory 5 (range 1 20))]
           (identical? (scal! 3 a) a) => true)
         (scal! 3 (gt factory 5 (range 1 20))) => (gt factory 5 (range 3 60 3))))

(defn test-gt-axpy [factory gt]
  (facts "BLAS 1 axpy! GT matrix"
         (with-release [a (gt factory 6 (range 1 100))
                        b (gt factory 6 (range 2 200 2))]
           (axpy -1 a a) => (gt factory 6)
           (identical? (axpy! 3.0 (gt factory 6 (range 60)) a) a) => true
           (axpy! 2.0 (gt factory 5 (range 1 70)) a) => (throws ExceptionInfo))))

(defn test-gt-mv [factory gt]
  (facts "BLAS 2 GT mv!"
         (mv! 2.0 (gt factory 6 (range 1 100))
              (vctr factory 1 2 3 4 5) 3 (vctr factory [1 2 3 4])) => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3 4 5])]
           (identical? (mv! 1.0 (gt factory 5 (range 1 100)) (vctr factory 1 2 3 4 5) 3 y) y))
         => true

         (with-release [y1 (vctr factory [1 2 3 4 5])
                        y2 (copy y1)
                        t (gt factory 5 (range 1 100))
                        g (let-release [res (ge factory 5 5)]
                            (copy! (dia t) (dia res))
                            (copy! (dia t 1) (dia res 1))
                            (copy! (dia t -1) (dia res -1))
                            res)]
           (mv! 1.0 t (vctr factory 1 2 3 4 5) 1.0 y2) => (mv! 1.0 g (vctr factory 1 2 3 4 5) 1.0 y1)
           (mv! t y1) => (throws ExceptionInfo))))

(defn test-gt-mm [factory gt]
  (facts "BLAS 3 GT mm!"
         (with-release [t (gt factory 5 (range 1 100))
                        g (let-release [res (ge factory 5 5)]
                            (copy! (dia t) (dia res))
                            (copy! (dia t 1) (dia res 1))
                            (copy! (dia t -1) (dia res -1))
                            res)]
           (mm! 2.0 t (ge factory 5 3 (range 1 16)))
           => (throws ExceptionInfo)

           (mm! 2.0 (trans t) (trans (gt factory 5 (range 1 100))))
           => (throws ExceptionInfo)

           (with-release [c (ge factory 5 2 (range 1 13))]
             (mm! 1.0 t (ge factory 5 2 (range 100)) 1.0 c)
             => (mm! 1.0 t (ge factory 5 2 (range 100)) 1.0 c)))))

(defn test-gt-entry! [factory gt]
  (facts "GT matrix entry!."
         (with-release [a (gt factory 5 (range 1 100))]
           (entry a 1 2) => 7.0
           (entry a 4 1) => 0.0
           (entry (entry! a 2 1 88.0) 2 1) => 88.0
           (entry! a 0 4 3.0) => (throws ExceptionInfo))))

(defn test-gt-alter! [factory gt]
  (facts "GT alter!."
         (entry (alter! (gt factory 5 (range 100)) 2 3 val+) 2 3) => 8.0
         (alter! (gt factory 4 (range 100)) val-ind+) => (throws ExceptionInfo)))

(defn test-gt-dot [factory gt]
  (facts "BLAS 1 GT dot"
         (with-release [a (gt factory 5 (range -10 100))
                        b (gt factory 5 (range 1 100))
                        d (gt factory 5)
                        zero-point (gt factory 5)]
           (sqrt (dot a a)) => (roughly (nrm2 a) 0.000001)
           (dot a b) => -182.0
           (dot a (copy! b d)) => (dot a b)
           (dot zero-point zero-point) => 0.0)))

(defn test-gt-nrm2 [factory gt]
  (facts "BLAS 1 GT nrm2."
         (with-release [a (gt factory 5 (repeat 1.0))]
           (nrm2 a) => (roughly (sqrt (dot a a)) 0.00001))
         (nrm2 (gt factory 1 [])) => 0.0))

(defn test-gt-amax [factory gt]
  (facts "BLAS 1 GT amax."
         (amax (gt factory 5 (range 1 100))) => 13.0
         (amax (gt factory 0 [])) => 0.0))

(defn test-gt-asum [factory gt]
  (facts "BLAS 1 GT asum."
         (asum (gt factory 5 (range 1 100))) => 91.0
         (asum (gt factory 0 [])) => 0.0))

(defn test-gt-sum [factory gt]
  (facts "BLAS 1 GT sum."
         (sum (gt factory 5 (range 1 100))) => 91.0))

(defn test-st-mv [factory]
  (facts "BLAS 2 ST mv!"
         (mv! 2.0 (st factory 5 (range 1 100))
              (vctr factory 1 2 3 4 5) 3 (vctr factory [1 2 3 4]))=> (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3 4 5])]
           (identical? (mv! 1.0 (st factory 5 (range 1 100)) (vctr factory 1 2 3 4 5) 1.0 y) y))
         => true

         (with-release [y1 (vctr factory [1 2 3 4 5])
                        y2 (copy y1)
                        t (st factory 5 (range 1 100))
                        g (let-release [res (ge factory 5 5)]
                            (copy! (dia t) (dia res))
                            (copy! (dia t 1) (dia res 1))
                            (copy! (dia t 1) (dia res -1))
                            res)]
           (mv! 1.0 t (vctr factory 1 2 3 4 5) 1.0 y2) => (mv! 1.0 g (vctr factory 1 2 3 4 5) 1.0 y1)
           (mv! t y1) => (throws ExceptionInfo))))

(defn test-st-mm [factory]
  (facts "BLAS 3 ST mm!"
         (with-release [t (st factory 5 (range 1 100))
                        g (let-release [res (ge factory 5 5)]
                            (copy! (dia t) (dia res))
                            (copy! (dia t 1) (dia res 1))
                            (copy! (dia t 1) (dia res -1))
                            res)]
           (mm! 2.0 t (ge factory 5 3 (range 1 16)))
           => (throws ExceptionInfo)

           (mm! 2.0 (trans t) (trans (st factory 5 (range 1 100))))
           => (throws ExceptionInfo)

           (with-release [c (ge factory 5 2 (range 1 13))]
             (mm! 1.0 t (ge factory 5 2 (range 100)) 1.0 c)
             => (mm! 1.0 t (ge factory 5 2 (range 100)) 1.0 c)))))

(defn test-st-constructor [factory]
  (facts "Create a ST matrix."
         (with-release [a (st factory 4 (range 1 100))]
           (st factory 4 nil) => (zero a))))

(defn test-st [factory]
  (facts "ST Matrix methods."
         (with-release [d (st factory 4 (range 1 100))]
           (dia d -2) => (vctr factory 0)
           (dia d) => (vctr factory [1 2 3 4])
           (dia d 1) => (vctr factory [5 6 7])
           (dia d -1) => (vctr factory 0)
           (trans (trans d)) => d)))

(defn test-st-entry! [factory]
  (facts "ST matrix entry!."
         (with-release [a (st factory 5 (range 1 100))]
           (entry a 1 2) => 7.0
           (entry a 2 1) => 7.0
           (entry a 4 1) => 0.0
           (entry (entry! a 1 2 88.0) 2 1) => 88.0
           (entry! a 0 4 3.0) => (throws ExceptionInfo))))

(defn test-st-alter! [factory]
  (facts "ST alter!."
         (entry (alter! (st factory 5 (range 100)) 2 3 val+) 3 2) => 8.0
         (alter! (st factory 4 (range 100)) val-ind+) => (throws ExceptionInfo)))

(defn test-st-dot [factory]
  (facts "BLAS 1 ST dot"
         (with-release [a (st factory 5 (range -10 100))
                        b (st factory 5 (range 1 100))
                        d (st factory 5)
                        zero-point (st factory 5)]
           (sqrt (dot a a)) => (roughly (nrm2 a) 0.000001)
           (dot a b) => -310.0
           (dot a (copy! b d)) => (dot a b)
           (dot zero-point zero-point) => 0.0)))

(defn test-st-amax [factory]
  (facts "BLAS 1 ST amax."
         (amax (st factory 5 (range 1 100))) => 9.0
         (amax (st factory 0 [])) => 0.0))

(defn test-st-asum [factory]
  (facts "BLAS 1 ST asum."
         (asum (st factory 5 (range 1 100))) => 75.0
         (asum (st factory 0 [])) => 0.0))

(defn test-st-sum [factory]
  (facts "BLAS 1 ST sum."
         (sum (st factory 5 (range 1 100))) => 75.0))

;; ==================== LAPACK tests =======================================

(defn test-vctr-srt [factory]
  (facts
   "LAPACK vector srt!"
   (with-release [x0 (vctr factory 0)
                  x1 (vctr factory [-1 1 2 3])
                  x2 (vctr factory [-1 1 2 3])
                  x3 (vctr factory [2 1 3 -1])
                  x4 (vctr factory [3 2 1 -1])
                  a (ge factory 2 2)]
     (sort+! x0) => x0
     (sort+! x1) => x2
     (sort-! x3) => x4
     (sort+! (row a 0)) => (throws ExceptionInfo))))

(defn test-ge-srt [factory]
  (facts
   "LAPACK ge srt!"
   (with-release [a0 (ge factory 2 0)
                  a1 (ge factory 2 3 [0 -1 1 2 3 3])
                  a2 (ge factory 2 3 [-1 0 1 2 3 3])
                  a3 (ge factory 2 3 [0 -1 2 1 3 3])]
     (sort+! a0) => a0
     (sort+! a1) => a2
     (sort-! a1) => a3)))

(defn test-ge-swap-rows [factory]
  (facts
   "LAPACK ge laswp!"
   (with-release [a (ge factory 2 3 [0 -1 1 2 3 4])
                  b (ge factory 2 3 [-1 0 2 1 4 3])
                  ipiv (vctr (index-factory factory) 2 1)]
     (swap-rows! a ipiv 1 1) => b)))

(defn test-uplo-srt [factory uplo]
  (facts
   "LAPACK uplo srt!"
   (with-release [a0 (uplo factory 0)
                  a1 (uplo factory 3 [0 -1 1 2 3 3])
                  a2 (uplo factory 3 [-1 0 1 2 3 3])
                  a3 (uplo factory 3 [1 0 -1 3 2 3])]
     (sort+! a0) => a0
     (sort+! a1) => a2
     (sort-! a1) => a3)))

(defn test-gb-srt [factory]
  (facts
   "LAPACK GB srt!"
   (with-release [a0 (gb factory 2 2 0 0)
                  a1 (gb factory 2 3 1 1 [3 0 -1 1 2])
                  a2 (gb factory 2 3 1 1 [0 3 -1 1 2])
                  a3 (gb factory 2 3 1 1 [3 0 1 -1 2])]
     (sort+! a0) => a0
     (sort+! a1) => a2
     (sort-! a1) => a3)))

(defn test-banded-uplo-srt [factory uplo]
  (facts
   "LAPACK banded uplo srt!"
   (with-release [a0 (uplo factory 0)
                  a1 (uplo factory 3 [0 -1 1 2 3 3])
                  a2 (uplo factory 3 [-1 0 1 2 3 3])
                  a3 (uplo factory 3 [1 0 -1 3 2 3])]
     (sort+! a0) => a0
     (sort+! a1) => a2
     (sort-! a1) => a3)))

(defn test-gt-srt [factory gt]
  (facts
   "LAPACK GT srt!"
   (with-release [a0 (gt factory 0)
                  a1 (gt factory 3 [0 -1 1 -5 -6 -7 -8])
                  a2 (gt factory 3 [-8 -7 -6 -5 -1 0 1])
                  a3 (gt factory 3 [1 0 -1 -5 -6 -7 -8])]
     (sort+! a0) => a0
     (sort+! a1) => a2
     (sort-! a1) => (sort-! a3))))

(defn test-gd-srt [factory]
  (facts
   "LAPACK GD srt!"
   (with-release [a0 (gd factory 0)
                  a1 (gd factory 3 [0 -1 1])
                  a2 (tb factory 3 0 [-1 0 1])
                  a3 (tb factory 3 0 [1 0 -1])]
     (sort+! a0) => a0
     (dia (sort+! a1)) => (dia a2)
     (dia (sort-! a1)) => (dia a3))))

(defn test-st-srt [factory gt]
  (facts
   "LAPACK ST srt!"
   (with-release [a0 (st factory 0)
                  a1 (st factory 3 [0 -1 1 -5 -6])
                  a2 (st factory 3 [-6 -5 -1 0 1])
                  a3 (st factory 3 [1 0 -1 -5 -6])]
     (sort+! a0) => a0
     (sort+! a1) => a2
     (sort-! a1) => (sort-! a3))))

(defn test-ge-trf [factory]
  (facts
   "LAPACK GE trf!"

   (with-release [a (ge factory 5 5 [6.80, -2.11,  5.66,  5.97,  8.23,
                                     -6.05, -3.30,  5.36, -4.44,  1.08,
                                     -0.45,  2.58, -2.70,  0.27,  9.04,
                                     8.32,  2.71,  4.35, -7.17,  2.14,
                                     -9.67, -5.14, -7.26,  6.08, -6.87])
                  ipiv (vctr (index-factory factory) 5)
                  lu (ge factory 5 5 [8.23   1.08   9.04   2.14  -6.87
                                      0.83  -6.94  -7.92   6.55  -3.99
                                      0.69  -0.67 -14.18   7.24  -5.19
                                      0.73   0.75   0.02 -13.82  14.19
                                      -0.26   0.44  -0.59  -0.34  -3.43]
                         {:layout :row})
                  lu-a (trf! a)]

     (:ipiv lu-a) => (vctr (index-factory factory) [5 5 3 4 5])
     (nrm2 (axpy! -1 a lu)) => (roughly 0 0.02))))

(defn test-sy-trx [factory sy]
  (facts
   "LAPACK SY linear system solver."

   (with-release [a (sy factory 8 [3
                                   5 3
                                   -2 2 0
                                   2 -2 0 8
                                   3 5 -2 -6 12
                                   -5 -3 2 -10 6 16
                                   -2 2 0 -8 8 8 6
                                   -3 -5 6 -14 6 20 18 34]
                        {:layout :row})
                  a-col (sy factory 8 [3
                                       5 3
                                       -2 2 0
                                       2 -2 0 8
                                       3 5 -2 -6 12
                                       -5 -3 2 -10 6 16
                                       -2 2 0 -8 8 8 6
                                       -3 -5 6 -14 6 20 18 34]
                            {:layout :column :uplo :upper})
                  b (ge factory 8 3 [1  -38  47
                                     7  -10  73
                                     6   52   2
                                     -30 -228 -42
                                     32  183 105
                                     34  297   9
                                     32  244  44
                                     62  497  61]
                        {:layout :row})
                  b-solution (ge factory 8 3 [1 1 8
                                              1 2 7
                                              1 3 6
                                              1 4 5
                                              1 5 4
                                              1 6 3
                                              1 7 2
                                              1 8 1]
                                 {:layout :row})
                  ipiv (vctr (index-factory factory) -2 -2 3 4 -6 -6 7 8)
                  ldl-a (sy factory 8 [3
                                       5 3
                                       1 -1 4
                                       -1 1 -1 8
                                       1 0 0 -1 1
                                       0 -1 1 -1 3 1
                                       1 -1 1 -1 -1 1 2
                                       -1 0 1 -1 1 0 1 16]
                            {:layout :row})
                  a-ge (transfer! a (ge factory 8 8))
                  ldl (trf a)
                  ldl-col (trf a-col)]
     (nrm2 (axpy! -1 b-solution (sv a b))) => (roughly 0 0.0001)
     (:ipiv ldl) => ipiv
     (con ldl (nrm1 a)) => (roughly 0 0.01)
     (asum (mm (tri ldl) a-ge)) => (roughly 8)
     (nrm2 (axpy! -1 (:lu ldl) ldl-a)) => (roughly 0 0.00001)
     (nrm2 (axpy! -1 b-solution (trs ldl-col b))) => (throws ExceptionInfo)
     (nrm2 (axpy! -1 b-solution (trs! ldl b))) => (roughly 0 0.0001))))

(defn test-sy-potrx [factory]
  (facts
   "LAPACK SY positive definite linear system solver."

   (with-release [a (sy factory 9 [1
                                   1 2
                                   1 2 3
                                   1 2 3 4
                                   1 2 3 4 5
                                   1 2 3 4 5 6
                                   1 2 3 4 5 6 7
                                   1 2 3 4 5 6 7 8
                                   1 2 3 4 5 6 7 8 9]
                        {:layout :row})
                  b (ge factory 9 2 [9 17 24 30 35 39 42 44 45
                                     45 89 131 170 205 235 259 276 285])
                  b-solution (ge factory 9 2 [1 1
                                              1 2
                                              1 3
                                              1 4
                                              1 5
                                              1 6
                                              1 7
                                              1 8
                                              1 9]
                                 {:layout :row})
                  inv (let [s (sy factory 9)]
                        (entry! (dia s) 2)
                        (entry! s 8 8 1)
                        (entry! (dia s -1) -1)
                        s)
                  gg (trf a)]
     (con gg (nrm1 a)) => (roughly 0 0.01)
     (nrm2 (axpy! -1 b-solution (trs gg b))) => (roughly 0 0.0001)
     (nrm2 (axpy! -1 (tri gg) inv)) => (roughly 0)
     (nrm2 (axpy! -1 b-solution (sv a b))) => (roughly 0 0.0001))))

(defn test-sp-potrx [factory]
  (facts
   "LAPACK SP positive definite linear system solver."

   (with-release [a (sp factory 9 [1
                                   1 2
                                   1 2 3
                                   1 2 3 4
                                   1 2 3 4 5
                                   1 2 3 4 5 6
                                   1 2 3 4 5 6 7
                                   1 2 3 4 5 6 7 8
                                   1 2 3 4 5 6 7 8 9]
                        {:layout :row})
                  b (ge factory 9 2 [9 17 24 30 35 39 42 44 45
                                     45 89 131 170 205 235 259 276 285])
                  b-solution (ge factory 9 2 [1 1
                                              1 2
                                              1 3
                                              1 4
                                              1 5
                                              1 6
                                              1 7
                                              1 8
                                              1 9]
                                 {:layout :row})
                  gg-sy (trf! (sy factory 9 [1
                                             1 2
                                             1 2 3
                                             1 2 3 4
                                             1 2 3 4 5
                                             1 2 3 4 5 6
                                             1 2 3 4 5 6 7
                                             1 2 3 4 5 6 7 8
                                             1 2 3 4 5 6 7 8 9]
                                  {:layout :row}))
                  gg (trf a)]
     (con gg (nrm1 a)) => (roughly (con gg-sy (nrm1 a)) 0.01)
     (nrm2 (axpy! -1 b-solution (trs gg b))) => (roughly 0 0.0001)
     (nrm2 (axpy! -1 (tri gg) (transfer! (tri gg-sy) (sp factory 9 {:layout :row})))) => (roughly 0)
     (nrm2 (axpy! -1 b-solution (sv a b))) => (roughly 0 0.0001))))

(defn test-gb-trx [factory]
  (facts
   "LAPACK GB linear system solver."

   (with-release [lu-a (let [res (gb factory 9 9 2 5)]
                         (entry! (dia res) 1)
                         (entry! (dia res -1) 2)
                         (entry! (dia res -2) 3)
                         (entry! (dia res 1) 2)
                         (entry! (dia res 2) 3)
                         (entry! (dia res 3) 4)
                         res)
                  a (subband lu-a 2 3)
                  b (ge factory 9 3 [1 1 1 1 1 1 1 1 1
                                     1 -1 1 -1 1 -1 1 -1 1
                                     1 2 3 4 5 6 7 8 9])
                  b-solution (ge factory 9 3 [0.629 -0.025 0.523 -0.286 -0.104 -0.118 0.220 -0.079 0.496
                                              1.149 1.870 0.615 -1.434 -0.524 -0.591 -0.896 1.604 0.480
                                              4.713 -0.726 2.946 -2.774 -1.067 -0.970 2.027 -0.340 3.599])
                  ipiv (vctr (index-factory factory) 3 4 5 6 5 6 9 8 9)
                  lu-solution (gb factory 9 9 2 5 [3 0.666 0.333
                                                   2 3 0.444 -0.111
                                                   1 2 3 0.518 0.692
                                                   2 1 2 3 0.567 0.246
                                                   3 2 1 2 -3.617 -0.334 -0.829
                                                   4 3 2 1 -4.419 -5.095 0.326 -0.588
                                                   4 3 2 -4.691 -3.174 3 0.043 -0.617
                                                   4 3 -4.074 -4.177 2 -1.546 -0.790
                                                   4 -2.271 -1.747 1 0.927 3.037])
                  lu (trf! a)]

     (:ipiv lu) => ipiv
     (nrm2 (axpy! -1 lu-a lu-solution)) => (roughly 0 0.1)
     (con lu (nrm1 a)) => (roughly 0 0.041)
     (nrm2 (axpy! -1 b-solution (trs! lu b))) => (roughly 0 0.004))))

(defn test-sb-trx [factory]
  (facts
   "LAPACK Symmetric banded linear system solver."

   (with-release [a (sb factory 9 3 [1.0 1.0 1.0 1.0
                                     2.0 2.0 2.0 1.0
                                     3.0 3.0 2.0 1.0
                                     4.0 3.0 2.0 1.0
                                     4.0 3.0 2.0 1.0
                                     4.0 3.0 2.0 1.0
                                     4.0 3.0 2.0
                                     4.0 3.0
                                     4.0])
                  a-copy (copy a)
                  b (ge factory 9 3 [4.0  0.0  1.0
                                     8.0  0.0  1.0
                                     12.0  0.0  0.0
                                     16.0  0.0  1.0
                                     16.0  0.0  0.0
                                     16.0  0.0 -1.0
                                     15.0  1.0  0.0
                                     13.0  1.0 -2.0
                                     10.0  2.0 -3.0]
                        {:layout :row})
                  b-solution (ge factory 9 3 [1.0  1.0  1.0
                                              1.0 -1.0  0.0
                                              1.0  1.0 -1.0
                                              1.0 -1.0  1.0
                                              1.0  1.0  0.0
                                              1.0 -1.0 -1.0
                                              1.0  1.0  1.0
                                              1.0 -1.0  0.0
                                              1.0  1.0 -1.0]
                                 {:layout :row})
                  gg-solution (doto (sb factory 9 3) (entry! 1.0))
                  gg (trf! a)]

     (nrm2 (axpy! -1 a gg-solution)) => (roughly 0 0.1)
     (con gg (nrm1 a)) => (roughly 0 0.041)
     (nrm2 (axpy! -1 b-solution (trs gg b))) => (roughly 0 0.004)
     (nrm2 (axpy! -1 (sv a-copy b) (trs gg b))) => (roughly 0 0.004))))

(defn test-tb-trx [factory]
  (facts
   "LAPACK triangular banded linear system solver."

   (with-release [a (tb factory 5 4 [6.80,
                                     -6.05, -3.30,
                                     -0.45,  2.58, -2.70,
                                     8.32,  2.71,  4.35, -7.17,
                                     -9.67, -5.14, -7.26,  6.08, -6.87])
                  t (tr factory 5 [6.80,
                                   -6.05, -3.30,
                                   -0.45,  2.58, -2.70,
                                   8.32,  2.71,  4.35, -7.17,
                                   -9.67, -5.14, -7.26,  6.08, -6.87])
                  b (ge factory 5 3 [4.02,  6.19, -8.22, -7.57, -3.03,
                                     -1.56,  4.00, -8.67,  1.75,  2.86,
                                     9.81, -4.09, -4.57, -8.61,  8.99])]
     (nrm2 (axpy! -1 (trs a b) (trs t b))) => (roughly 0.0 0.0001)
     (con a) => (roughly (con t) 0.0000001))))

(defn test-gd-trx [factory]
  (facts
   "LAPACK diagonal linear system solver."

   (with-release [a (gd factory 5 [1 2 3 4 5])
                  b (ge factory 5 2 [1 2 3 4 5 1 1 1 1 1])
                  t (tb factory 5 0 [1 2 3 4 5])
                  b-row (ge factory 5 2 [1 1 2 1 3 1 4 1 5 1] {:layout :row})
                  b-solution (ge factory 5 2 [1 1 1 1 1 1 1/2 1/3 1/4 1/5])]
     (nrm2 (axpy! -1 b-solution (trs a b) )) => (roughly 0.0 0.0001)
     (nrm2 (axpy! -1 b-solution (sv a b))) => (roughly 0.0 0.0001)
     (nrm2 (axpy! -1 b-solution (sv a b-row) )) => (roughly 0.0 0.0001)
     (con a) => (roughly (con t) 0.0000001))))

(defn test-gt-trx [factory]
  (facts
   "LAPACK GT linear system solver."

   (with-release [a (let [res (gt factory 9)]
                      (entry! (dia res) 1)
                      (entry! (dia res 1) 4)
                      (entry! (dia res -1) 3)
                      res)
                  lu-d (vctr factory [3.000,  3.666, 3.000,  3.303, 3.532,  3.000, 4.799,  3.000,  4.332])
                  lu-du (vctr factory [1.000, -1.333, 1.000, -2.787, 4.000,  1.000, 3.196,  1.000])
                  lu-dl (vctr factory [0.333,  0.818, 0.696,  0.908, 0.849, -0.799, 0.625, -0.332])
                  ipiv (vctr (index-factory factory) [2 2 4 4 5 7 7 9 9])
                  b (ge factory 9 3 [1 1 1 1 1 1 1 1 1
                                     1 -1 1 -1 1 -1 1 -1 1
                                     1 2 3 4 5 6 7 8 9])
                  b-solution (ge factory 9 3 [0.609   0.478   4.597
                                              0.098   0.130  -0.899
                                              -0.231  -0.641  -2.723
                                              0.234   0.312   2.105
                                              0.364   0.153   2.516
                                              -0.017  -0.023  -0.958
                                              -0.019  -0.359  -0.147
                                              0.267   0.357   2.505
                                              0.198  -0.070   1.484]
                                 {:layout :row})
                  lu (trf a)]

     (:ipiv lu) => ipiv
     (nrm2 (axpy! -1 (dia (:lu lu)) lu-d)) => (roughly 0 0.002)
     (nrm2 (axpy! -1 (dia (:lu lu) -1) lu-dl)) => (roughly 0 0.002)
     (nrm2 (axpy! -1 (dia (:lu lu) 1) lu-du)) => (roughly 0 0.002)
     (con lu 8) => (roughly 0.03667)
     (nrm2 (axpy! -1 b-solution (trs! lu b))) => (roughly 0 0.002))))

(defn test-dt-trx [factory]
  (facts
   "LAPACK DT linear system solver."

   (with-release [a (let [res (dt factory 9)]
                      (entry! (dia res) 1)
                      (entry! (dia res 1) 4)
                      (entry! (dia res -1) 3)
                      res)
                  ldl (trf a)]
     (:lu ldl) => truthy)))

(defn test-st-trx [factory]
  (facts
   "LAPACK PT linear system solver."

   (with-release [a (let [res (dt factory 9)]
                      (entry! (dia res) 1)
                      (entry! (dia res 1) 4)
                      res)
                  ldl (trf a)]

     (:lu ldl) => truthy)))

(defn test-ge-trs [factory]
  (facts
   "LAPACK GE trs!"

   (with-release [a (ge factory 5 5 [6.80, -2.11,  5.66,  5.97,  8.23,
                                     -6.05, -3.30,  5.36, -4.44,  1.08,
                                     -0.45,  2.58, -2.70,  0.27,  9.04,
                                     8.32,  2.71,  4.35, -7.17,  2.14,
                                     -9.67, -5.14, -7.26,  6.08, -6.87])
                  b (ge factory 5 3 [4.02,  6.19, -8.22, -7.57, -3.03,
                                     -1.56,  4.00, -8.67,  1.75,  2.86,
                                     9.81, -4.09, -4.57, -8.61,  8.99])
                  ipiv (vctr (index-factory factory) 5)
                  solution (ge factory 5 3 [-0.80  -0.39   0.96
                                            -0.70  -0.55   0.22
                                            0.59   0.84   1.90
                                            1.32  -0.10   5.36
                                            0.57   0.11   4.04]
                               {:layout :row})
                  lu (ge factory 5 5 [8.23   1.08   9.04   2.14  -6.87
                                      0.83  -6.94  -7.92   6.55  -3.99
                                      0.69  -0.67 -14.18   7.24  -5.19
                                      0.73   0.75   0.02 -13.82  14.19
                                      -0.26   0.44  -0.59  -0.34  -3.43]
                         {:layout :row})
                  lu-a (trf! a)]

     (:ipiv lu-a) => (vctr (index-factory factory) [5 5 3 4 5])
     (trs! lu-a b) => b
     (nrm2 (axpy! -1 a lu)) => (roughly 0.0136)
     (nrm2 (axpy! -1 b solution)) => (roughly 0.0120))))

(defn test-tr-trs [factory tr]
  (facts
   "LAPACK TR trs!"

   (with-release [t (tr factory 5 [6.80,
                                   -6.05, -3.30,
                                   -0.45,  2.58, -2.70,
                                   8.32,  2.71,  4.35, -7.17,
                                   -9.67, -5.14, -7.26,  6.08, -6.87])
                  b (ge factory 5 3 [4.02,  6.19, -8.22, -7.57, -3.03,
                                     -1.56,  4.00, -8.67,  1.75,  2.86,
                                     9.81, -4.09, -4.57, -8.61,  8.99])
                  a (transfer! t (ge factory 5 5))
                  b1 (copy b)
                  lu-a (trf! a)]

     (nrm2 (axpy! -1 (trs! t b) (trs! lu-a b1))) => (roughly 0.0 0.0001))))

(defn test-ge-tri [factory]
  (facts
   "LAPACK GE trf/tri"
   (with-release [a (ge factory 5 5 [0.378589,   0.971711,   0.016087,   0.037668,   0.312398,
                                     0.756377,   0.345708,   0.922947,   0.846671,   0.856103,
                                     0.732510,   0.108942,   0.476969,   0.398254,   0.507045,
                                     0.162608,   0.227770,   0.533074,   0.807075,   0.180335,
                                     0.517006,   0.315992,   0.914848,   0.460825,   0.731980])
                  lu-a (trf! (copy a))]

     (nrm2 (mm (tri! lu-a) a)) => (roughly (sqrt 5.0)))))

(defn test-tr-tri [factory tr]
  (facts
   "LAPACK TR tri"
   (with-release [a (ge factory 5 5 [0.378589, 0 0 0 0
                                     0.756377,   0.345708, 0 0 0
                                     0.732510,   0.108942,   0.476969, 0 0
                                     0.162608,   0.227770,   0.533074,   0.807075, 0
                                     0.517006,   0.315992,   0.914848,   0.460825,   0.731980]
                        {:layout :row})
                  inv-a (tri! (copy (view-tr a)))]
     (nrm2 (mm inv-a a)) => (roughly (sqrt 5.0)))))

(defn test-ge-sv [factory]
  (facts
   "LAPACK GE sv!"

   (with-release [a (ge factory 5 5 [6.80, -2.11,  5.66,  5.97,  8.23,
                                     -6.05, -3.30,  5.36, -4.44,  1.08,
                                     -0.45,  2.58, -2.70,  0.27,  9.04,
                                     8.32,  2.71,  4.35, -7.17,  2.14,
                                     -9.67, -5.14, -7.26,  6.08, -6.87])
                  b (ge factory 5 3 [4.02,  6.19, -8.22, -7.57, -3.03,
                                     -1.56,  4.00, -8.67,  1.75,  2.86,
                                     9.81, -4.09, -4.57, -8.61,  8.99])
                  ipiv (vctr (index-factory factory) 5)
                  solution (ge factory 5 3 [-0.80  -0.39   0.96
                                            -0.70  -0.55   0.22
                                            0.59   0.84   1.90
                                            1.32  -0.10   5.36
                                            0.57   0.11   4.04]
                               {:layout :row})
                  lu (ge factory 5 5 [8.23   1.08   9.04   2.14  -6.87
                                      0.83  -6.94  -7.92   6.55  -3.99
                                      0.69  -0.67 -14.18   7.24  -5.19
                                      0.73   0.75   0.02 -13.82  14.19
                                      -0.26   0.44  -0.59  -0.34  -3.43]
                         {:layout :row})]

     (nrm2 (axpy! -1 (sv! a b) solution)) => (roughly 0.01201)
     (nrm2 (axpy! -1 a lu)) => (roughly 0.01359))))

(defn test-tr-sv [factory tr]
  (facts
   "LAPACK TR sv!"

   (with-release [t (tr factory 5 [6.80,
                                   -6.05, -3.30,
                                   -0.45,  2.58, -2.70,
                                   8.32,  2.71,  4.35, -7.17,
                                   -9.67, -5.14, -7.26,  6.08, -6.87])
                  b (ge factory 5 3 [4.02,  6.19, -8.22, -7.57, -3.03,
                                     -1.56,  4.00, -8.67,  1.75,  2.86,
                                     9.81, -4.09, -4.57, -8.61,  8.99])
                  t1 (copy t)
                  b1 (copy b)
                  t2 (transfer! t (tr factory 5 {:layout :row}))
                  b2 (copy b)
                  t3 (copy t2)
                  b3 (copy b)]

     (sv! t b) => (trs! t1 b1)
     (nrm2 (axpy! -1 (sv! t2 b2) b)) => (roughly 0 0.0001)
     (nrm2 (axpy! -1 (sv! t3 b3) b2)) => (roughly 0 0.0001))))

(defn test-sy-sv [factory]
  (facts
   "LAPACK SY sv!"

   (with-release [a (sy factory 5 [6.80,
                                   -6.05, -3.30,
                                   -0.45,  2.58, -2.70,
                                   8.32,  2.71,  4.35, -7.17,
                                   -9.67, -5.14, -7.26,  6.08, -6.87])
                  b (ge factory 5 3 [4.02,  6.19, -8.22, -7.57, -3.03,
                                     -1.56,  4.00, -8.67,  1.75,  2.86,
                                     9.81, -4.09, -4.57, -8.61,  8.99])]

     (nrm2 (axpy! -1 (mm a (sv a b)) b)) => (roughly 0 0.0001))))

(defn test-ge-det [factory]
  (facts
   "LAPACK GE det"

   (with-release [a (ge factory 5 5 [6.80, -2.11,  5.66,  5.97,  8.23,
                                     -6.05, -3.30,  5.36, -4.44,  1.08,
                                     -0.45,  2.58, -2.70,  0.27,  9.04,
                                     8.32,  2.71,  4.35, -7.17,  2.14,
                                     -9.67, -5.14, -7.26,  6.08, -6.87])
                  lu (ge factory 5 5 [8.23   1.08   9.04   2.14  -6.87
                                      0.83  -6.94  -7.92   6.55  -3.99
                                      0.69  -0.67 -14.18   7.24  -5.19
                                      0.73   0.75   0.02 -13.82  14.19
                                      -0.26   0.44  -0.59  -0.34  -3.43]
                         {:layout :row})]

     (det (trf! a)) => (roughly -38406.4848))))

(defn test-ge-con [factory]
  (facts
   "LAPACK GE con!"

   (with-release [a (ge factory 4 4  [1.80 2.88 2.05 -0.89
                                      5.25 -2.95 -0.95 -3.80
                                      1.58 -2.69 -2.90 -1.04
                                      -1.11 -0.66 -0.59 0.80]
                        {:layout :row})
                  lu-a (trf a)
                  nrm (nrm1 a)
                  copy-a (copy a)
                  lu-copy-a (trf! copy-a)]

     (con lu-a nrm true) => (roughly (/ 152.1620))
     (con lu-copy-a) => (throws ExceptionInfo))))

(defn test-tr-con [factory tr]
  (facts
   "LAPACK TR con!"

   (with-release [a (ge factory 4 4  [1.80 0 0 0
                                      5.25 -2.95 0 0
                                      1.58 -2.69 -2.90 0
                                      -1.11 -0.66 -0.59 0.80]
                        {:layout :row})
                  b (copy (view-tr a))
                  lu (trf a)
                  nrm (nrm1 a)]

     (con b) => (roughly (con lu nrm)))))

(defn test-ge-qr [factory]
  (facts
   "LAPACK GE qrf"

   (with-release [a (ge factory 6 4 [ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
                                     -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
                                     -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
                                     4.53,  3.83, -6.64,  2.06, -2.47,  4.70])
                  c (ge factory 6 2 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  qr-solution (ge factory 6 4 [-17.54  -4.76  -1.96   0.42
                                               -0.52  12.40   7.88  -5.84
                                               -0.40  -0.14  -5.75   4.11
                                               0.44  -0.66  -0.20  -7.78
                                               0.37  -0.26  -0.17  -0.15
                                               -0.29   0.46   0.41   0.24]
                                  {:layout :row})
                  q-solution (ge factory 6 4 [-0.08  -0.66  -0.12  -0.15
                                              0.57   0.20   0.64  -0.27
                                              0.43   0.43  -0.65   0.21
                                              -0.48   0.47  -0.11  -0.70
                                              -0.40   0.05   0.08   0.30
                                              0.31  -0.34  -0.37  -0.52]
                                 {:layout :row})
                  orm-result (ge factory 6 2 [-7.65   2.25
                                              13.53   4.63
                                              0.24   2.79
                                              2.15  -6.31
                                              -1.86  -2.29
                                              1.42   5.73]
                                 {:layout :row})
                  qr (qrf a)]
     (nrm2 (axpy! -1 (:or qr) qr-solution)) => (roughly 0 0.015)
     (nrm2 (axpy! -1 (org qr) q-solution)) => (roughly 0 0.014)
     (nrm2 (axpy! -1 (mm! qr c) orm-result)) => (roughly 0 0.012))))

(defn test-ge-qp [factory]
  (facts
   "LAPACK GE qpf"

   (with-release [a (ge factory 6 4 [ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
                                     -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
                                     -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
                                     4.53,  3.83, -6.64,  2.06, -2.47,  4.70])
                  c (ge factory 6 2 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  qr-solution (ge factory 6 4 [-17.540 -4.7556 0.422672 -1.9595
                                               -0.52475 12.3982 -5.8428 7.87891
                                               -0.39778 -0.14081 8.79755 -2.6836
                                               0.439403 -0.66066 -0.66481 -5.0799
                                               0.373018 -0.25674 0.079529 -0.27499
                                               -0.28714 0.460641 -0.03549 0.547166]
                                  {:layout :row})
                  qr (qpf a)]

     (nrm2 (axpy! -1 (:or qr) qr-solution)) => (roughly 0 0.015))))

(defn test-ge-rq [factory]
  (facts
   "LAPACK GE rqf"

   (with-release [a (ge factory 4 6 [1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
                                     -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
                                     -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
                                     4.53,  3.83, -6.64,  2.06, -2.47,  4.70]
                        {:layout :row})
                  c (ge factory 6 4 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44
                                     8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  rq-solution (ge factory 4 6 [-0.32   0.82  16.01  -5.88   4.02   0.70
                                               -0.18   0.36  -0.44  -7.78   8.13   7.04
                                               0.26   0.16  -0.36  -0.80   7.38   6.67
                                               0.30   0.25  -0.43   0.13  -0.16 -10.57]
                                  {:layout :row})
                  q-solution (ge factory 4 6 [0.31  -0.73  -0.41   0.20   0.38  -0.17
                                              0.40  -0.41   0.45  -0.28  -0.23   0.58
                                              -0.21  -0.11   0.28   0.89  -0.11   0.24
                                              -0.43  -0.36   0.63  -0.19   0.23  -0.44]
                                 {:layout :row})
                  orm-result (ge factory 6 4 [27778.10474205729 34849.97134137374 27778.10474205729 34849.97134137374
                                              37576.96048773454 47101.054569347405 37576.96048773454 47101.054569347405
                                              3883.645144131808 4867.867395762253 3883.645144131808 4867.867395762253
                                              -650.254385760667 -804.37260305241 -650.254385760667 -804.37260305241
                                              -35.23575939807661 -43.69759691590371 -35.23575939807661 -43.69759691590371
                                              -4824.495361548854 -6051.555558128556 -4824.495361548854 -6051.555558128556]
                                 {:layout :row})
                  rq (rqf a)]
     (nrm2 (axpy! -1 (:or rq) rq-solution)) => (roughly 0 0.015)
     (nrm2 (axpy! -1 (org rq) q-solution)) => (roughly 0 0.014)
     (nrm2 (axpy! -1 (mm! rq c) orm-result)) => (roughly 0 0.04))))

(defn test-ge-lq [factory]
  (facts
   "LAPACK GE lqf"

   (with-release [a (ge factory 4 4 [2 3 -1 0
                                     -6 -5 0 2
                                     2 -5 6 -6
                                     4 6 2 -3]
                        {:layout :row})
                  c (ge factory 4 4 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44
                                     8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  lq-solution (ge factory 4 4 [-3.74   0.52  -0.17   0.00
                                               7.22  -3.60  -0.42   0.36
                                               4.54   8.84  -1.49  -0.80
                                               -6.41   3.81  -2.30   2.00]
                                  {:layout :row})
                  q-solution (ge factory 4 4 [ -0.534522 -0.801784 0.267261 -0
                                              0.595961 -0.218519 0.536365 -0.55623
                                              0.564904 -0.386513 -0.0297318 0.728428
                                              -0.2 0.4 0.8 0.4]
                                 {:layout :row})
                  orm-result (ge factory 4 4 [3861.9576278420636 4161.270425791217 1103.4822844002706 3861.9576278420636
                                              27719.90284478883 29911.457438130125 7937.11129835215 27719.90284478883
                                              16655.246287501766 17979.057772570286 4748.23322679897 16655.246287501766
                                              -25024.58322860647 -27015.7721311638 -7195.2447564577 -25024.58322860647]
                                 {:layout :row})
                  lq (lqf a)]

     (nrm2 (axpy! -1 (:or lq) lq-solution)) => (roughly 0 0.015)
     (nrm2 (axpy! -1 (org lq) q-solution)) => (roughly 0 0.014)
     (nrm2 (axpy! -1 (mm! lq c) orm-result)) => (roughly 0 0.012))))

(defn test-ge-ql [factory]
  (facts
   "LAPACK GE qlf"

   (with-release [a (ge factory 4 4 [2 3 -1 0
                                     -6 -5 0 2
                                     2 -5 6 -6
                                     4 6 2 -3]
                        {:layout :row})
                  c (ge factory 4 4 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44
                                     8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  ql-solution (ge factory 4 4 [-0.56  -0.23  -0.29  -0.00
                                               -0.41  -4.53   0.47  -0.20
                                               5.75   8.62  -2.24   0.60
                                               -5.14   0.29  -6.00   7.00 ]
                                  {:layout :row})
                  q-solution (ge factory 4 4 [0.87   0.19   0.45   0.00
                                              0.47  -0.34  -0.77   0.29
                                              0.13   0.32  -0.38  -0.86
                                              0.06  -0.86   0.26  -0.43]
                                 {:layout :row})
                  orm-result (ge factory 4 4 [73364.78125 54602.0859375 6721.794921875 73364.78125
                                              -23043.171875 -17142.7265625 -2109.498779296875 -23043.171875
                                              6087.68798828125 4543.7509765625 562.00439453125 6087.68798828125
                                              134.61383056640625 98.04141998291016 12.774694442749023 134.61383056640625]
                                 {:layout :row})
                  ql (qlf a)]

     (nrm2 (axpy! -1 (:or ql) ql-solution)) => (roughly 0 0.015)
     (nrm2 (axpy! -1 (org ql) q-solution)) => (roughly 0 0.014)
     (nrm2 (axpy! -1 (mm! ql c) orm-result)) => (roughly 0 0.14))))

(defn test-ge-ls [factory]
  (facts
   "LAPACK GE ls!"

   (with-release [a (ge factory 6 4 [ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
                                     -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
                                     -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
                                     4.53,  3.83, -6.64,  2.06, -2.47,  4.70])
                  b (ge factory 6 2 [8.58,  8.26,  8.48, -5.28,  5.72,  8.93,
                                     9.35, -4.43, -0.70, -0.26, -7.36, -2.52])
                  solution (ge factory 4 2 [-0.45   0.25
                                            -0.85  -0.90
                                            0.71   0.63
                                            0.13   0.14]
                               {:layout :row})
                  qr (ge factory 6 4 [-17.54  -4.76  -1.96   0.42
                                      -0.52  12.40   7.88  -5.84
                                      -0.40  -0.14  -5.75   4.11
                                      0.44  -0.66  -0.20  -7.78
                                      0.37  -0.26  -0.17  -0.15
                                      -0.29   0.46   0.41   0.24]
                         {:layout :row})
                  residual (vctr factory [195.36 107.06])]
     (ls! a b)
     (nrm2 (axpy! -1 residual (ls-residual a b))) => (roughly 0.003 0.0001)
     (nrm2 (axpy! -1 solution (submatrix b 0 0 4 2))) 0.007 => (roughly 0.007)
     (nrm2 (axpy! -1 qr a)) => (roughly 0.0129))))

(defn test-ge-lse [factory]
  (facts
   "LAPACK GE lse!"

   (with-release [a (ge factory 6 4 [-0.57 -1.28 -0.39 0.25
                                     -1.93 1.08 -0.31 -2.14
                                     2.30 0.24 0.40 -0.35
                                     -1.93 0.64 -0.66 0.08
                                     0.15 0.30 0.15 -2.13
                                     -0.02 1.03 -1.43 0.50]
                        {:layout :row})
                  b (ge factory 2 4 [1 0 -1 0
                                     0 1 0 -1]
                        {:layout :row})
                  c (vctr factory [-1.50 -2.14 1.23 -0.54 -1.68 0.82])
                  d (vctr factory [0 0])
                  x (vctr factory 4)
                  solution (vctr factory [0.4890 0.9975 0.4890 0.9975])
                  residual 0.0251]
     (nrm2 (axpy! -1 (lse! a b c d x) solution)) => (roughly 0.0 0.0001))))

(defn test-ge-gls [factory]
  (facts
   "LAPACK GE gls!"

   (with-release [a (ge factory 4 3 [-0.57 -1.28 -0.39
                                     -1.93 1.08 -0.31
                                     2.30 0.24 -0.40
                                     -0.02 1.03 -1.43]
                        {:layout :row})
                  b (ge factory 4 4 [0.5 0 0 0
                                     0 1 0 0
                                     0 0 2 0
                                     0 0 0 5]
                        {:layout :row})
                  d (vctr factory [1.32 -4 5.52 3.24])
                  x-solution (vctr factory [1.98887 -1.0058 -2.9911])]
     (nrm2 (axpy! -1 (get (gls a b d) 0) x-solution)) => (roughly 0.0 0.0001))))

(defn test-ge-ev [factory]
  (facts
   "LAPACK GE ev!"

   (with-release [a0 (ge factory 5 5 [-1.01,  3.98,  3.30,  4.43,  7.31,
                                      0.86,  0.53,  8.26,  4.96, -6.43,
                                      -4.60, -7.04, -3.89, -7.66, -6.16,
                                      3.31,  5.29,  8.20, -7.33,  2.47,
                                      -4.81,  3.55, -1.51,  6.18,  5.58])
                  a1 (copy a0)
                  w (ge factory 5 2)
                  eigenvalues (ge factory 5 2 [2.86  10.76
                                               2.86 -10.76
                                               -0.69   4.70
                                               -0.69  -4.70
                                               -10.46   0.00]
                                  {:layout :row})
                  vl-res (ge factory 5 5 [-0.04  -0.29  -0.13  -0.33   0.04
                                          -0.62   0.00   0.69   0.00   0.56
                                          0.04   0.58  -0.39  -0.07  -0.13
                                          -0.28  -0.01  -0.02  -0.19  -0.80
                                          0.04  -0.34  -0.40   0.22   0.18]
                             {:layout :row})
                  vr-res (ge factory 5 5 [-0.11  -0.17  -0.73   0.00   0.46
                                          -0.41   0.26   0.03   0.02   0.34
                                          -0.10   0.51  -0.19   0.29   0.31
                                          -0.40   0.09   0.08   0.08  -0.74
                                          -0.54   0.00   0.29   0.49   0.16]
                             {:layout :row})
                  vl (ge factory 5 5)
                  vr (ge factory 5 5)]

     (ev! a1 w vl vr) = truthy
     (nrm2 (axpy! -1 eigenvalues (ev! a0 w))) => (roughly 0.00943)
     (nrm2 (axpy! -1 (fmap! abs vl-res) (fmap! abs vl))) => (roughly 0.01378)
     (nrm2 (axpy! -1 (fmap! abs vr-res) (fmap! abs vr))) => (roughly 0.010403))))

(defn test-sy-evd [factory]
  (facts
   "LAPACK SY ev! with syevd"

   (with-release [s (sy factory 5 [-1.01,
                                   3.98, 0.53,,
                                   3.30, 8.26, -3.89,
                                   4.43, 4.96, -7.66, -7.33,
                                   7.31, -6.43, -6.16, 2.47,  5.58])
                  w-res (ge factory 5 1 [-16.90 -12.11 -5.92 7.56 14.57])
                  v-res (ge factory 5 5 [0.20   -0.48   -0.53    0.59   -0.32
                                         -0.33    0.63   -0.38    0.47    0.36
                                         -0.58   -0.49    0.44    0.36    0.32
                                         0.58    0.23    0.57    0.53    0.05
                                         -0.41    0.29    0.24    0.15   -0.81]
                            {:layout :row})
                  w (ge factory 5 1)
                  v (ge factory 5 5)]

     (nrm2 (axpy! -1 w-res (ev! s w nil v))) => (roughly 0 0.01)
     (nrm2 (axpy! -1 v-res v)) => (roughly 0 0.02))))

(defn test-sy-evr [factory]
  (facts
   "LAPACK SY ev! with syevr algorithm"

   (with-release [s (sy factory 5 [ 0.67
                                   -0.20   3.82
                                   0.19  -0.13   3.27
                                   -1.06   1.06   0.11   5.86
                                   0.46  -0.48   1.10  -0.98   3.54]
                        {:layout :row})
                  w-res (ge factory 3 1 [0.43   2.14   3.37])
                  z-res (ge factory 5 3 [-0.98  -0.01  -0.08
                                         0.01   0.02  -0.93
                                         0.04  -0.69  -0.07
                                         -0.18   0.19   0.31
                                         0.07   0.69  -0.13]
                            {:layout :row})
                  w (ge factory 3 1)
                  z (ge factory 5 3 {:layout :row})]

     (nrm2 (axpy! -1 w-res (ev! s w nil z))) => (roughly 0 0.007)
     (nrm2 z) => (roughly (nrm2 z-res) 0.01))))

(defn test-ge-es [factory]
  (facts
   "LAPACK GE es!"

   (with-release [a0 (ge factory 4 4 [ 0.35   0.45  -0.14  -0.17
                                      0.09   0.07  -0.54   0.35
                                      -0.44  -0.33  -0.03   0.17
                                      0.25  -0.32  -0.13   0.11]
                         {:layout :row})
                  a1 (copy a0)
                  a2 (copy a0)
                  w0 (ge factory 4 2)
                  w1 (ge factory 4 2)
                  vs (ge factory 4 4 {:layout :row})
                  eigenvalues (ge factory 4 2 [2.86  10.76
                                               2.86 -10.76
                                               -0.69   4.70
                                               -0.69  -4.70]
                                  {:layout :row})
                  t-res (ge factory 4 4 [0.7995 -0.0059 -0.0751 -0.0927
                                         0.0000 -0.1007  0.3937  0.3569
                                         0.0000  0.0000 -0.0994 -0.5128
                                         0.0000  0.0000  0.3132 -0.0994]
                            {:layout :row})
                  vs-res (ge factory 4 4 [-0.6551 -0.1210 -0.5032  0.5504
                                          -0.5236 -0.3286  0.7857  0.0229
                                           0.5362 -0.5974  0.0904  0.5894
                                          -0.0956 -0.7215 -0.3482 -0.5908]
                             {:layout :row})]

     (es! a0 w0 vs) => (ev! a1 w1)
     (nrm2 (axpy! -1 (mm vs a0 (trans vs)) a2)) => (roughly 0.0 0.000001))))

(defn test-ge-svd [factory]
   (with-release [a (ge factory 6 5 [8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
                                     9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
                                     9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
                                     5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
                                     3.16,  7.98,  3.01,  5.80,  4.27, -5.31])
                  s-res (gd factory 5 [27.47 22.64 8.56 5.99 2.01])
                  u-res (ge factory 6 5 [-0.59   0.26   0.36   0.31   0.23
                                         -0.40   0.24  -0.22  -0.75  -0.36
                                         -0.03  -0.60  -0.45   0.23  -0.31
                                         -0.43   0.24  -0.69   0.33   0.16
                                         -0.47  -0.35   0.39   0.16  -0.52
                                         0.29   0.58  -0.02   0.38  -0.65]
                            {:layout :row})
                  vt-res (ge factory 5 5 [-0.25  -0.40  -0.69  -0.37  -0.41
                                          0.81   0.36  -0.25  -0.37  -0.10
                                          -0.26   0.70  -0.22   0.39  -0.49
                                          0.40  -0.45   0.25   0.43  -0.62
                                          -0.22   0.14   0.59  -0.63  -0.44]
                             {:layout :row})]
     (facts "LAPACK GE svd!"
      (with-release [a0 (copy a)
                     a1 (copy a0)
                     a2 (copy a0)
                     a3 (copy a0)
                     s (gd factory 5)
                     s0 (copy s)
                     u (ge factory 6 5)
                     vt (ge factory 5 5)
                     superb (gd factory 5)]
        (nrm2 (axpy! -1 s-res (svd! a0 s superb))) => (roughly 0.007526)
        (nrm2 (axpy! -1 s-res (svd! a1 s u vt superb))) => (roughly 0.007526)
        (nrm2 (axpy! -1 (svd! a2 s superb) (svd! a3 s0))) => (roughly 0)
        (nrm2 (axpy! -1 u-res u)) => (roughly 0.01586)
        (nrm2 (axpy! -1 vt-res vt)) => (roughly 0.012377)))

     (facts "LAPACK GE divide and conquer svd! (sdd)"
      (with-release [a0 (copy a)
                     a1 (copy a0)
                     a2 (copy a0)
                     a3 (copy a0)
                     s (gd factory 5)
                     s0 (copy s)
                     u (ge factory 6 5)
                     vt (ge factory 5 5)]

        (nrm2 (axpy! -1 s-res (svd! a0 s))) => (roughly 0.007526)
        (nrm2 (axpy! -1 s-res (svd! a1 s u vt))) => (roughly 0.007526)
        (svd! a2 s) => (:sigma (svd a3 true))
        (nrm2 (axpy! -1 u-res u)) => (roughly 0.01586)
        (nrm2 (axpy! -1 vt-res vt)) => (roughly 0.012377)))))

;; =========================================================================

(defn test-blas [factory]
  (test-group factory)
  (test-vctr-constructor factory)
  (test-vctr factory)
  (test-vctr-bulk-entry! factory)
  (test-vctr-dot factory)
  (test-sum factory)
  (test-iamax factory)
  (test-vctr-nrm2 factory)
  (test-vctr-asum factory)
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
  (test-ge-asum factory)
  (test-ge-dot factory)
  (test-ge-nrm2 factory)
  (test-ge-mv factory)
  (test-rk factory)
  (test-ge-mm factory)
  (test-tr-constructor factory)
  (test-tr factory)
  (test-uplo-copy factory tr)
  (test-uplo-swap factory tr)
  (test-uplo-scal factory tr)
  (test-uplo-axpy factory tr)
  (test-tr-mv factory tr)
  (test-tr-mm factory tr)
  (test-sy-constructor factory)
  (test-sy factory)
  (test-uplo-copy factory sy)
  (test-uplo-swap factory sy)
  (test-uplo-scal factory sy)
  (test-uplo-axpy factory sy)
  (test-sy-mv factory sy)
  (test-sy-rk factory sy)
  (test-sy-mmt factory sy)
  (test-sy-mm factory))

(defn test-blas-host [factory]
  (test-iamin factory)
  (test-imax factory)
  (test-imin factory)
  (test-rot factory)
  (test-rotg factory)
  (test-rotm factory)
  (test-rotmg factory)
  (test-sum-asum factory)
  (test-vctr-entry factory)
  (test-vctr-entry! factory)
  (test-vctr-alter! factory)
  (test-vctr-amax factory)
  (test-ge-entry factory)
  (test-ge-entry! factory)
  (test-ge-alter! factory)
  (test-ge-amax factory)
  (test-ge-sum factory)
  (test-ge-sum-asum factory)
  (test-ge-dot-strided factory)
  (test-ge-trans! factory)
  (test-tr-entry factory tr)
  (test-tr-entry! factory tr)
  (test-tr-bulk-entry! factory tr)
  (test-tr-alter! factory tr)
  (test-tr-dot factory tr)
  (test-tr-nrm2 factory tr)
  (test-tr-asum factory tr)
  (test-tr-sum factory tr)
  (test-tr-amax factory tr)
  (test-sy-rk factory sp))

(defn test-basic-int-host [factory]
  (test-vctr-swap factory)
  (test-vctr-copy factory)
  (test-vctr-int-entry factory)
  (test-vctr-int-entry! factory)
  (test-vctr-int-alter! factory))

(defn test-lapack [factory]
  (test-vctr-srt factory)
  (test-ge-srt factory)
  (test-ge-swap factory)
  (test-uplo-srt factory tr)
  (test-uplo-srt factory sy)
  (test-uplo-srt factory sp)
  (test-uplo-srt factory tp)
  (test-gb-srt factory)
  (test-banded-uplo-srt factory tb)
  (test-banded-uplo-srt factory sb)
  (test-gt-srt factory gt)
  (test-gt-srt factory dt)
  (test-gd-srt factory)
  (test-ge-trf factory)
  (test-ge-trs factory)
  (test-tr-trs factory tr)
  (test-tr-trs factory tp)
  (test-ge-tri factory)
  (test-tr-tri factory tr)
  (test-tr-tri factory tp)
  (test-ge-con factory)
  (test-tr-con factory tr)
  (test-tr-con factory tp)
  (test-ge-det factory)
  (test-ge-sv factory)
  (test-tr-sv factory tr)
  (test-tr-sv factory tp)
  (test-sy-sv factory)
  (test-sy-trx factory sy)
  (test-sy-potrx factory)
  (test-gb-trx factory)
  (test-sb-trx factory)
  (test-tb-trx factory)
  (test-sy-trx factory sp)
  (test-gt-trx factory)
  (test-sp-potrx factory)
  (test-gd-trx factory)
  (test-gt-trx factory)
  (test-dt-trx factory)
  (test-st-trx factory)
  (test-ge-qr factory)
  (test-ge-qp factory)
  (test-ge-rq factory)
  (test-ge-lq factory)
  (test-ge-ql factory)
  (test-ge-ls factory)
  (test-ge-lse factory)
  (test-ge-gls factory)
  (test-ge-ev factory)
  (test-sy-evd factory)
  (test-sy-evr factory)
  (test-ge-es factory)
  (test-ge-svd factory))

(defn test-blas-sy-host [factory]
  (test-sy-entry factory sy)
  (test-sy-entry! factory sy)
  (test-sy-bulk-entry! factory sy)
  (test-sy-alter! factory sy)
  (test-sy-dot factory)
  (test-sy-nrm2 factory sy)
  (test-sy-asum factory sy)
  (test-sy-sum factory sy)
  (test-sy-amax factory sy))

(defn test-blas-gb [factory]
  (test-gb-constructor factory)
  (test-gb factory)
  (test-gb-copy factory)
  (test-gb-swap factory)
  (test-gb-scal factory)
  (test-gb-axpy factory)
  (test-gb-mv factory)
  (test-gb-mm factory))

(defn test-blas-gb-host [factory]
  (test-gb-entry! factory)
  (test-gb-alter! factory)
  (test-gb-dot factory)
  (test-gb-nrm2 factory)
  (test-gb-asum factory)
  (test-gb-sum factory)
  (test-gb-amax factory))

(defn test-blas-sb [factory]
  (test-banded-uplo-constructor factory sb)
  (test-banded-uplo factory sb)
  (test-uplo-copy factory sb)
  (test-uplo-swap factory sb)
  (test-uplo-scal factory sb)
  (test-uplo-axpy factory sb)
  (test-sy-mv factory sb)
  (test-sy-mm factory))

(defn test-blas-sb-host [factory]
  (test-sb-entry factory)
  (test-sb-entry! factory)
  (test-sb-bulk-entry! factory)
  (test-sb-alter! factory)
  (test-sy-dot factory)
  (test-sy-nrm2 factory sb)
  (test-sy-asum factory sb)
  (test-sy-sum factory sb)
  (test-sy-amax factory sb))

(defn test-blas-tb [factory]
  (test-banded-uplo-constructor factory tb)
  (test-banded-uplo factory tb)
  (test-uplo-copy factory tb)
  (test-uplo-swap factory tb)
  (test-uplo-scal factory tb)
  (test-uplo-axpy factory tb)
  (test-tb-mv factory)
  (test-tb-mm factory))

(defn test-blas-tb-host [factory]
  (test-tb-entry factory)
  (test-tb-entry! factory)
  (test-tb-bulk-entry! factory)
  (test-tb-alter! factory)
  (test-tr-dot factory tr)
  (test-tr-nrm2 factory tr)
  (test-tr-asum factory tr)
  (test-tr-sum factory tr)
  (test-tr-amax factory tr))

(defn test-blas-tp [factory]
  (test-packed-constructor factory tp)
  (test-packed factory tp)
  (test-packed-copy factory tp)
  (test-uplo-swap factory tp)
  (test-uplo-scal factory tp)
  (test-uplo-axpy factory tp)
  (test-tr-mv factory tp)
  (test-tp-mm factory))

(defn test-blas-tp-host [factory]
  (test-tr-entry factory tp)
  (test-tr-entry! factory tp)
  (test-tr-bulk-entry! factory tp)
  (test-tr-alter! factory tp)
  (test-tp-dot factory)
  (test-tr-nrm2 factory tp)
  (test-tr-asum factory tp)
  (test-tr-sum factory tp)
  (test-tr-amax factory tp))

(defn test-blas-sp [factory]
  (test-packed-constructor factory sp)
  (test-packed factory sp)
  (test-packed-copy factory sp)
  (test-uplo-swap factory sp)
  (test-uplo-scal factory sp)
  (test-uplo-axpy factory sp)
  (test-sy-mv factory sp)
  (test-sp-mm factory))

(defn test-blas-sp-host [factory]
  (test-sy-entry factory sp)
  (test-sy-entry! factory sp)
  (test-sy-bulk-entry! factory sp)
  (test-sy-alter! factory sp)
  (test-sp-dot factory)
  (test-sy-nrm2 factory sp)
  (test-sy-asum factory sp)
  (test-sy-sum factory sp)
  (test-sy-amax factory sp))

(defn test-blas-gd [factory]
  (test-gd-constructor factory)
  (test-gd factory)
  (test-gd-copy factory)
  (test-gd-swap factory)
  (test-gd-scal factory)
  (test-gd-axpy factory)
  (test-gd-mv factory)
  (test-gd-mm factory))

(defn test-blas-gd-host [factory]
  (test-gd-entry! factory)
  (test-gd-alter! factory)
  (test-gd-dot factory)
  (test-gd-nrm2 factory)
  (test-gd-asum factory)
  (test-gd-sum factory)
  (test-gd-amax factory))

(defn test-blas-gt [factory]
  (test-gt-constructor factory gt)
  (test-gt factory gt)
  (test-gt-copy factory gt)
  (test-gt-swap factory gt)
  (test-gt-scal factory gt)
  (test-gt-axpy factory gt)
  (test-gt-mv factory gt)
  (test-gt-mm factory gt))

(defn test-blas-gt-host [factory]
  (test-gt-entry! factory gt)
  (test-gt-alter! factory gt)
  (test-gt-dot factory gt)
  (test-gt-nrm2 factory gt)
  (test-gt-asum factory gt)
  (test-gt-sum factory gt)
  (test-gt-amax factory gt))

(defn test-blas-dt [factory]
  (test-gt-constructor factory dt)
  (test-gt factory dt)
  (test-gt-copy factory dt)
  (test-gt-swap factory dt)
  (test-gt-scal factory dt)
  (test-gt-axpy factory dt)
  (test-gt-mv factory dt)
  (test-gt-mm factory dt))

(defn test-blas-dt-host [factory]
  (test-gt-entry! factory dt)
  (test-gt-alter! factory dt)
  (test-gt-dot factory dt)
  (test-gt-nrm2 factory dt)
  (test-gt-asum factory dt)
  (test-gt-sum factory dt)
  (test-gt-amax factory dt))

(defn test-blas-st [factory]
  (test-st-constructor factory)
  (test-st factory)
  (test-gt-copy factory st)
  (test-gt-swap factory st)
  (test-gt-scal factory st)
  (test-gt-axpy factory st)
  (test-st-mv factory)
  (test-st-mm factory))

(defn test-blas-st-host [factory]
  (test-st-entry! factory)
  (test-st-alter! factory)
  (test-st-dot factory)
  (test-gt-nrm2 factory st)
  (test-st-asum factory)
  (test-st-sum factory)
  (test-st-amax factory))
