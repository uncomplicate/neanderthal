;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.real-test
  (:require [midje.sweet :refer [facts throws => roughly truthy]]
            [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [linalg :refer :all]
             [aux :refer :all]
             [math :refer :all]
             [real :refer [ls-residual]]]
            [uncomplicate.neanderthal.internal.api :refer [data-accessor index-factory]])
  (:import clojure.lang.ExceptionInfo))

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
         (vctr? (vctr factory [1 2 3])) => true
         (vctr factory 1 2 3) => (vctr factory [1 2 3])
         (vctr factory 3) => (vctr factory [0 0 0])
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

;;================ BLAS functions =========================================

(defn test-vctr-dot [factory]
  (facts "BLAS 1 dot."
         (dot (vctr factory [1 2 3]) (vctr factory [10 30 -10])) => 40.0
         (dot (vctr factory [1 2 3]) nil) => (throws ExceptionInfo)
         (dot (vctr factory []) (vctr factory [])) => 0.0
         (dot (vctr factory [1]) (ge factory 3 3)) => (throws ClassCastException)
         (dot (vctr factory [1 2]) (vctr factory [1 2 3])) => (throws ExceptionInfo)))

(defn test-sum [factory]
  (facts "BLAS Plus 1 sum."
         (sum (vctr factory [1 2 -5]))  => -2.0
         (sum (vctr factory [])) => 0.0))

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
   (with-release [x (vctr factory [1 2 3])]
     (swp! x (vctr factory 3)) => x
     (swp! x (vctr factory [10 20 30])) => (vctr factory [10 20 30])
     (swp! x nil) => (throws ExceptionInfo)
     (identical? (swp! x (vctr factory [10 20 30])) x) => true
     (swp! x (vctr factory [10 20])) => (throws ExceptionInfo))))

(defn test-vctr-scal [factory]
  (facts "BLAS 1 vector scal!"
         (with-release [x (vctr factory [1 2 3])]()
           (identical? (scal! 3 x) x) => true)
         (scal! 3 (vctr factory [1 -2 3])) => (vctr factory [3 -6 9])
         (scal! 3 (vctr factory [])) => (vctr factory [])))

(defn test-vctr-copy [factory]
  (facts "BLAS 1 vector copy!"
         (with-release [y (vctr factory 3)]
           (identical? (copy! (vctr factory [1 2 3]) y) y) => true
           (identical? (copy y) y) => false
           (copy (vctr factory [1 2 3])) => y)

         (copy! (vctr factory [10 20 30]) (vctr factory [1 2 3]))
         => (vctr factory [10 20 30])

         (copy! (vctr factory [1 2 3]) nil) => (throws ExceptionInfo)
         (copy! (vctr factory [10 20 30]) (vctr factory [1])) => (throws ExceptionInfo)

         (copy (vctr factory 1 2)) => (vctr factory 1 2)))

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
                        v (vctr factory [1 2 5 6 9 10 13 14])
                        b (tr factory 2 [1 2 6])
                        c (ge factory 2 7 (range 1 15))]
           a1 => a
           (ge factory 2 4 nil) => (zero a)
           (view-vctr a) => v
           (view-ge a) => a
           (view-ge c 2) => a
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

         (trans (ge factory 2 3 (range 6))) => (ge factory 3 2 (range 6) {:order :row})))

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
         (entry (alter! (ge factory 2 3 [1 2 3 4 5 6] {:order :row}) 1 1 val+) 1 1) => 6.0
         (alter! (ge factory 2 3 [1 2 3 4 5 6]) val-ind+) => (ge factory 2 3 [0 0 3 5 10 14])
         (alter! (ge factory 2 3 [1 2 3 4 5 6] {:order :row}) val-ind+)
         => (ge factory 2 3 [0 2 6 0 6 14] {:order :row})))

(defn test-ge-dot [factory]
  (facts "BLAS 1 GE dot"
         (let [a (ge factory 2 3 (range -3 3))
               b (ge factory 2 3 [1 2 3 4 5 6])
               zero-point (ge factory 0 0 [])]
           (sqrt (dot a a)) => (roughly (nrm2 a))
           (dot a b) => (dot (view-vctr a) (view-vctr b))
           (dot zero-point zero-point) => 0.0)))

(defn test-ge-dot-strided [factory]
  (facts "BLAS 1 GE dot"
         (let [a (ge factory 2 3 (range -3 3))
               b (ge factory 2 3 [1 2 3 4 5 6])
               c (ge factory 3 4 (repeat 1))
               d (ge factory 2 3 {:order :row})
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
                        b-row (ge factory 2 3 {:order :row})]
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
                  b-row (ge factory 2 3 [7 9 11 8 10 12] {:order :row})]
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
                        b (ge factory 3 2 [0 3 1 4 2 5] {:order :row})]
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

         (mv! (ge factory 2 3 [1 3 5 2 4 6] {:order :row}) (vctr factory 1 2 3) (vctr factory 0 0))
         => (mv! (ge factory 2 3 [1 2 3 4 5 6] {:order :column}) (vctr factory 1 2 3) (vctr factory 0 0))

         (mv 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:order :row}) (vctr factory 1 2 3) (vctr factory 0 0))
         => (vctr factory 22 28)))

(defn test-rk [factory]
  (facts "BLAS 2 rk!"
         (with-release [a (ge factory 2 3)
                        b (ge factory 2 3 {:order :row})]
           (identical? (rk! 2.0 (vctr factory 1 2) (vctr factory 1 2 3) a) a)
           (identical? (rk! 2.0 (vctr factory 1 2) (vctr factory 1 2 3) b) b))
         => true

         (rk! (vctr factory 3 2 1 4) (vctr factory 1 2 3)
                (ge factory 4 3 [1 2 3 4 2 2 2 2 3 4 2 1]))
         => (ge factory 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

         (rk! (vctr factory 1 2) (vctr factory 1 2 3) (ge factory 2 2 [1 2 3 5]))
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

         (mm! 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:order :row}) (ge factory 3 2 [1 4 2 5 3 6] {:order :row})
              3.0  (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [47 62 107 140])

         (mm 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:order :row}) (ge factory 3 2 [1 4 2 5 3 6] {:order :row})
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
           (dia a-upper) => (vctr factory 0 2 5))))

(defn test-tr-entry [factory]
  (facts "TR Matrix entry."
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

(defn test-tr-entry! [factory]
  (facts "TR matrix entry!."
         (with-release [a (tr factory 3 [1 2 3 4 5 6] {:diag :unit :uplo :upper})
                        c (ge factory 3 3 (range 1 10))]
           (entry (entry! a 0 1 88.0) 0 1) => 88.0
           (entry (entry! a 1 0 3.0) 0 1) => (throws ExceptionInfo)
           (entry (entry! a 1 0 4.0) 1 1) => (throws ExceptionInfo)
           (entry (view-ge (entry! (view-tr c {:diag :unit}) -1)) 2 2) => 9.0)))

(defn test-tr-bulk-entry! [factory]
  (facts "Bulk TR matrix entry!."
         (sum (row (entry! (tr factory 3 [1 2 3 4 5 6]) 33.0) 1)) => 66.0
         (sum (row (entry! (tr factory 3 [1 2 3 4 5 6] {:diag :unit :uplo :upper}) 22.0) 0)) => 44.0))

(defn test-tr-alter! [factory]
  (facts "TR alter!."
         (entry (alter! (tr factory 3 [1 2 3 4 5 6] {:order :row}) 1 1 val+) 1 1) => 4.0
         (alter! (tr factory 3 [1 2 3 4 5 6]) val-ind+) => (tr factory 3 [0 0 0 5 7 16])
         (alter! (tr factory 3 [1 2 3 4 5 6] {:order :row}) val-ind+)
         => (tr factory 3 [0 0 4 0 7 16] {:order :row})))

(defn test-tr-dot [factory]
  (facts "BLAS 1 TR dot"
         (let [a (tr factory 3 (range -3 3))
               b (tr factory 3 [1 2 3 4 5 6])
               d (tr factory 3 {:order :row})
               zero-point (tr factory 0 [])]
           (sqrt (dot a a)) => (roughly (nrm2 a))
           (dot a b) => 7.0
           (dot a (copy! b d)) => (dot a b)
           (dot zero-point zero-point) => 0.0)))

(defn test-tr-nrm2 [factory]
  (facts "BLAS 1 TR nrm2."
         (nrm2 (tr factory 3 (range -5 5))) => (roughly (sqrt 55))
         (nrm2 (tr factory 0 [])) => 0.0))

(defn test-tr-amax [factory]
  (facts "BLAS 1 TR amax."
         (amax (tr factory 3 [1 2 3 -4 -3 1 1 1 1 1 1 1])) => 4.0
         (amax (tr factory 0 [])) => 0.0))

(defn test-tr-asum [factory]
  (facts "BLAS 1 TR asum."
         (asum (tr factory 3 (range -3 3))) => 9.0
         (asum (tr factory 0 [])) => 0.0))

(defn test-tr-sum [factory]
  (facts "BLAS 1 TR sum."
         (sum (tr factory 3 (range -3 3))) => -3.0
         (sum (tr factory 0 [])) => 0.0))

(defn test-uplo-copy [factory uplo uplo-name]
  (facts (format "BLAS 1 copy! %s matrix" uplo-name)
         (with-release [a (uplo factory 3)
                        b (uplo factory 3 (range 6) {:order :column})
                        b-row (uplo factory 3 (range 6) {:order :row})]

           (identical? (copy a) a) => false
           (identical? (copy! (uplo factory 3 [1 2 3 4 5 6]) a) a) => true
           (copy (uplo factory 3 [1 2 3 4 5 6])) => a
           (copy! b b-row) => b)

         (copy! (uplo factory 3 [10 20 30 40 50 60]) (uplo factory 3 [1 2 3 4 5 6]))
         => (uplo factory 3 [10 20 30 40 50 60])

         (copy! (uplo factory 3 [1 2 3 4 5 6]) nil) => (throws ExceptionInfo)
         (copy! (uplo factory 3 [10 20 30 40 50 60]) (uplo factory 2)) => (throws ExceptionInfo)))

(defn test-uplo-swap [factory uplo uplo-name]
  (facts
   (format "BLAS 1 swap! %s matrix" uplo-name)
   (with-release [a (uplo factory 3 [1 2 3 4 5 6])]
     (identical? (swp! a (uplo factory 3)) a) => true
     (swp! a (uplo factory 3 [10 20 30 40 50 60])) => (uplo factory 3 [10 20 30 40 50 60])
     (swp! a nil) => (throws ExceptionInfo)
     (identical? (swp! a (uplo factory 3 [10 20 30 40 50 60])) a) => true
     (swp! a (uplo factory 2 [10 20])) => (throws ExceptionInfo))))

(defn test-uplo-scal [factory uplo uplo-name]
  (facts (format "BLAS 1 scal! %s matrix" uplo-name)
         (with-release [a (uplo factory 3 [1 2 3 4 5 6])]
           (identical? (scal! 3 a) a) => true)
         (scal! 3 (uplo factory 3 [1 -2 3 9 8 7])) => (uplo factory 3 [3 -6 9 27 24 21])))

(defn test-uplo-axpy [factory uplo uplo-name]
  (facts (format "BLAS 1 axpy! %s matrix" uplo-name)
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

(defn test-tr-mv [factory]
  (facts "BLAS 2 TR mv!"
         (mv! 2.0 (tr factory 2 [1 2 3 4])
              (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4])) => (throws ExceptionInfo)

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
           (view-sy b) => a
           (view-ge (view-sy b)) => b
           (sy factory 3 nil) => (zero a))))

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
           (dia a-upper) => (vctr factory 0 2 5))))

(defn test-sy-mv [factory]
  (facts "BLAS 2 SY mv!"
         (mv! 2.0 (sy factory 2 [1 2 3 4]) (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4]))
         => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3])]
           (identical? (mv! 2 (sy factory 3 [1 2 3 4 5 6]) (vctr factory 1 2 3) 3 y) y)) => true

         (mv! 3 (sy factory 3 [1 10 2 20 3 30] {:order :row}) (vctr factory 7 0 4) 3 (vctr factory 3))
         => (vctr factory 261 246 780)

         (mv! 2 (sy factory 3 [1 2 3 4 5 6] {:uplo :upper}) (vctr factory 1 2 3) 3 (vctr factory 3))
         => (vctr factory 34 46 64)

         (mv! (sy factory 3 [1 2 4 3 5 6] {:order :row}) (vctr factory 1 2 3) (vctr factory 3))
         => (mv! (sy factory 3 [1 2 3 4 5 6] {:order :column}) (vctr factory 1 2 3) (vctr factory 3))))

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

(defn test-sy-entry [factory]
  (facts "SY Matrix entry."
         (with-release [a-upper (sy factory 3 (range 6) {:uplo :upper})]
           (entry a-upper 0 1) => 1.0
           (entry a-upper 1 0) => 1.0
           (entry a-upper 1 1) => 2.0
           (entry a-upper 2 0) => 3.0
           (entry a-upper 0 2) => 3.0
           (entry a-upper -1 2) => (throws ExceptionInfo)
           (entry a-upper 3 2) => (throws ExceptionInfo))))

(defn test-sy-entry! [factory]
  (facts "SY matrix entry!."
         (with-release [a (sy factory 3 [1 2 3 4 5 6] {:uplo :upper})
                        c (ge factory 3 3 (range 1 10))]

           (entry (view-ge (entry! (view-sy c) -1)) 1 2) => 8.0
           (entry (entry! a 0 1 88.0) 0 1) => 88.0
           (entry (entry! a 1 0 3.0) 0 1) => (throws ExceptionInfo)
           (entry (entry! a 1 0 4.0) 1 1) => (throws ExceptionInfo))))

(defn test-sy-bulk-entry! [factory]
  (facts "Bulk SY matrix entry!."
         (sum (row (entry! (sy factory 3 [1 2 3 4 5 6]) 33.0) 1)) => 66.0
         (sum (row (entry! (sy factory 3 [1 2 3 4 5 6] {:uplo :upper}) 22.0) 0)) => 66.0))

(defn test-sy-alter! [factory]
  (facts "SY alter!."
         (entry (alter! (sy factory 3 [1 2 3 4 5 6] {:order :row}) 1 1 val+) 1 1) => 4.0
         (alter! (sy factory 3 [1 2 3 4 5 6]) val-ind+) => (sy factory 3 [0 0 0 5 7 16])
         (alter! (sy factory 3 [1 2 3 4 5 6] {:order :row}) val-ind+)
         => (sy factory 3 [0 0 4 0 7 16] {:order :row})))

(defn test-sy-dot [factory]
  (facts "BLAS 1 SY dot"
         (let [a (sy factory 3 (range -3 3))
               b (sy factory 3 [1 2 3 4 5 6])
               d (sy factory 3 {:order :row})
               zero-point (sy factory 0 [])]
           (sqrt (dot a a)) => (roughly (nrm2 a))
           (dot a b) => 5.0
           (dot a (copy! b d)) => (dot a b)
           (dot zero-point zero-point) => 0.0)))

(defn test-sy-nrm2 [factory]
  (facts "BLAS 1 SY nrm2."
         (nrm2 (sy factory 3 (range -5 5))) => (roughly (sqrt 81))
         (nrm2 (sy factory 0 [])) => 0.0))

(defn test-sy-amax [factory]
  (facts "BLAS 1 SY amax."
         (amax (sy factory 3 [1 2 3 -4 -3 1 1 1 1 1 1 1])) => 4.0
         (amax (sy factory 0 [])) => 0.0))

(defn test-sy-asum [factory]
  (facts "BLAS 1 SY asum."
         (asum (sy factory 3 (range -3 3))) => 13.0
         (asum (sy factory 0 [])) => 0.0))

(defn test-sy-sum [factory]
  (facts "BLAS 1 SY sum."
         (sum (sy factory 3 (range -3 3))) => -5.0
         (sum (sy factory 0 [])) => 0.0))

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

(defn test-tr-srt [factory]
  (facts
   "LAPACK ge srt!"
   (with-release [a0 (tr factory 0)
                  a1 (tr factory 3 [0 -1 1 2 3 3])
                  a2 (tr factory 3 [-1 0 1 2 3 3])
                  a3 (tr factory 3 [1 0 -1 3 2 3])]
     (sort+! a0) => a0
     (sort+! a1) => a2
     (sort-! a1) => a3)))

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
                         {:order :row})
                  lu-a (trf! a)]

     (:ipiv lu-a) => (vctr (index-factory factory) [5 5 3 4 5])
     (nrm2 (axpy! -1 a lu)) => (roughly 0.01359))))

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
                               {:order :row})
                  lu (ge factory 5 5 [8.23   1.08   9.04   2.14  -6.87
                                      0.83  -6.94  -7.92   6.55  -3.99
                                      0.69  -0.67 -14.18   7.24  -5.19
                                      0.73   0.75   0.02 -13.82  14.19
                                      -0.26   0.44  -0.59  -0.34  -3.43]
                         {:order :row})
                  lu-a (trf! a)]

     (:ipiv lu-a) => (vctr (index-factory factory) [5 5 3 4 5])
     (trs! lu-a b) => b
     (nrm2 (axpy! -1 a lu)) => (roughly 0.0136)
     (nrm2 (axpy! -1 b solution)) => (roughly 0.0120))))

(defn test-tr-trs [factory]
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
                  a (copy (view-ge t))
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

(defn test-tr-tri [factory]
  (facts
   "LAPACK TR tri"
   (with-release [a (ge factory 5 5 [0.378589, 0 0 0 0
                                     0.756377,   0.345708, 0 0 0
                                     0.732510,   0.108942,   0.476969, 0 0
                                     0.162608,   0.227770,   0.533074,   0.807075, 0
                                     0.517006,   0.315992,   0.914848,   0.460825,   0.731980]
                        {:order :row})
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
                               {:order :row})
                  lu (ge factory 5 5 [8.23   1.08   9.04   2.14  -6.87
                                      0.83  -6.94  -7.92   6.55  -3.99
                                      0.69  -0.67 -14.18   7.24  -5.19
                                      0.73   0.75   0.02 -13.82  14.19
                                      -0.26   0.44  -0.59  -0.34  -3.43]
                         {:order :row})]

     (nrm2 (axpy! -1 (sv! a b) solution)) => (roughly 0.01201)
     (nrm2 (axpy! -1 a lu)) => (roughly 0.01359))))

(defn test-tr-sv [factory]
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
                  t2 (copy! t (tr factory 5 {:order :row}))
                  b2 (copy b)
                  t3 (copy t2)
                  b3 (copy b)]

     (sv! t b) => (trs! t1 b1)
     (nrm2 (axpy! -1 (sv! t2 b2) b)) => (roughly 0 0.0001)
     (nrm2 (axpy! -1 (sv! t3 b3) b2)) => (roughly 0 0.0001))))

(defn test-ge-det [factory]
  (facts
   "LAPACK GE det"

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
                         {:order :row})]

     (det (trf! a)) => (roughly -38406.4848))))

(defn test-ge-con [factory]
  (facts
   "LAPACK GE con!"

   (with-release [a (ge factory 4 4  [1.80 2.88 2.05 -0.89
                                      5.25 -2.95 -0.95 -3.80
                                      1.58 -2.69 -2.90 -1.04
                                      -1.11 -0.66 -0.59 0.80]
                        {:order :row})
                  lu-a (trf a)
                  copy-a (copy a)
                  lu-copy-a (trf! copy-a)]

     (con lu-a) => (roughly (/ 152.1620))
     (con lu-copy-a) => (throws ExceptionInfo))))

(defn test-tr-con [factory]
  (facts
   "LAPACK TR con!"

   (with-release [a (ge factory 4 4  [1.80 0 0 0
                                      5.25 -2.95 0 0
                                      1.58 -2.69 -2.90 0
                                      -1.11 -0.66 -0.59 0.80]
                        {:order :row})
                  b (copy (view-tr a))
                  lu (trf a)]

     (con b) => (roughly (con lu)))))

(defn test-ge-qr [factory]
  (facts
   "LAPACK GE qr!"

   (with-release [a0 (ge factory 6 4 [ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
                                      -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
                                      -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
                                      4.53,  3.83, -6.64,  2.06, -2.47,  4.70])
                  a1 (copy a0)
                  tau (vctr factory 4)
                  c (ge factory 6 2 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  qr (ge factory 6 4 [-17.54  -4.76  -1.96   0.42
                                      -0.52  12.40   7.88  -5.84
                                      -0.40  -0.14  -5.75   4.11
                                      0.44  -0.66  -0.20  -7.78
                                      0.37  -0.26  -0.17  -0.15
                                      -0.29   0.46   0.41   0.24]
                         {:order :row})
                  q (ge factory 6 4 [-0.08  -0.66  -0.12  -0.15
                                      0.57   0.20   0.64  -0.27
                                      0.43   0.43  -0.65   0.21
                                      -0.48   0.47  -0.11  -0.70
                                      -0.40   0.05   0.08   0.30
                                      0.31  -0.34  -0.37  -0.52]
                        {:order :row})
                  qr*c (ge factory 6 2 [-7.65   2.25
                                        13.53   4.63
                                        0.24   2.79
                                        2.15  -6.31
                                        -1.86  -2.29
                                        1.42   5.73]
                           {:order :row})]
     (nrm2 (axpy! -1 (gqr! a0 (qrf! a0 tau)) q)) => (roughly 0.014)
     (nrm2 (axpy! -1 (doto a1 (qrf! tau)) qr)) => (roughly 0.0129)
     (nrm2 (axpy! -1 (mqr! a1 tau c) qr*c)) => (roughly 0.0115))))

(defn test-ge-rq [factory]
  (facts
   "LAPACK GE rq!"

   (with-release [a0 (ge factory 4 6 [ 1.44, -9.96, -7.55,  8.34,  7.08, -5.45,
                                      -7.84, -0.28,  3.24,  8.09,  2.52, -5.70,
                                      -4.39, -3.24,  6.27,  5.28,  0.74, -1.19,
                                      4.53,  3.83, -6.64,  2.06, -2.47,  4.70]
                         {:order :row})
                  a1 (copy a0)
                  tau (vctr factory 4)
                  c (ge factory 6 4 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44
                                     8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  rq (ge factory 4 6 [-0.32   0.82  16.01  -5.88   4.02   0.70
                                      -0.18   0.36  -0.44  -7.78   8.13   7.04
                                      0.26   0.16  -0.36  -0.80   7.38   6.67
                                      0.30   0.25  -0.43   0.13  -0.16 -10.57]
                         {:order :row})
                  q (ge factory 4 6 [0.31  -0.73  -0.41   0.20   0.38  -0.17
                                     0.40  -0.41   0.45  -0.28  -0.23   0.58
                                     -0.21  -0.11   0.28   0.89  -0.11   0.24
                                     -0.43  -0.36   0.63  -0.19   0.23  -0.44]
                        {:order :row})
                  rq*c (ge factory 6 4 [27778.10474205729 34849.97134137374 27778.10474205729 34849.97134137374
                                        37576.96048773454 47101.054569347405 37576.96048773454 47101.054569347405
                                        3883.645144131808 4867.867395762253 3883.645144131808 4867.867395762253
                                        -650.254385760667 -804.37260305241 -650.254385760667 -804.37260305241
                                        -35.23575939807661 -43.69759691590371 -35.23575939807661 -43.69759691590371
                                        -4824.495361548854 -6051.555558128556 -4824.495361548854 -6051.555558128556]
                           {:order :row})]
     (nrm2 (axpy! -1 (grq! a0 (rqf! a0 tau)) q)) => (roughly 0.01304)
     (nrm2 (axpy! -1 (doto a1 (rqf! tau)) rq)) => (roughly 0.0127)
     (nrm2 (axpy! -1 (mrq! a1 tau c) rq*c)) => (roughly 0.0 0.04))))

(defn test-ge-lq [factory]
  (facts
   "LAPACK GE lq!"

   (with-release [a0 (ge factory 4 4 [2 3 -1 0
                                      -6 -5 0 2
                                      2 -5 6 -6
                                      4 6 2 -3]
                         {:order :row})
                  a1 (copy a0)
                  a2 (copy a0)
                  tau (vctr factory 4)
                  c (ge factory 4 4 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44
                                     8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  lq (ge factory 4 4 [-3.74   0.52  -0.17   0.00
                                      7.22  -3.60  -0.42   0.36
                                      4.54   8.84  -1.49  -0.80
                                      -6.41   3.81  -2.30   2.00]
                         {:order :row})
                  q (ge factory 4 4 [ -0.534522 -0.801784 0.267261 -0
                                     0.595961 -0.218519 0.536365 -0.55623
                                     0.564904 -0.386513 -0.0297318 0.728428
                                     -0.2 0.4 0.8 0.4]
                        {:order :row})
                  lq*c (ge factory 4 4 [3861.9576278420636 4161.270425791217 1103.4822844002706 3861.9576278420636
                                        27719.90284478883 29911.457438130125 7937.11129835215 27719.90284478883
                                        16655.246287501766 17979.057772570286 4748.23322679897 16655.246287501766
                                        -25024.58322860647 -27015.7721311638 -7195.2447564577 -25024.58322860647]
                           {:order :row})]

     (nrm2 (axpy! -1 (glq! a0 (lqf! a0 tau)) q)) => (roughly 0.0 0.0001)
     (nrm2 (axpy! -1 (doto a1 (lqf! tau)) lq)) => (roughly 0.0126)
     (nrm2 (axpy! -1 (mlq! a1 tau c) lq*c)) => (roughly 0.0 0.01))))

(defn test-ge-ql [factory]
  (facts
   "LAPACK GE ql!"

   (with-release [a0 (ge factory 4 4 [2 3 -1 0
                                      -6 -5 0 2
                                      2 -5 6 -6
                                      4 6 2 -3]
                         {:order :row})
                  a1 (copy a0)
                  a2 (copy a0)
                  tau (vctr factory 4)
                  c (ge factory 4 4 [8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44
                                     8.58,  8.26,  8.48, -5.28, 2.3 2.2
                                     9.35, -4.43, -0.70, -0.26 2.1 0.44])
                  ql (ge factory 4 4 [-0.56  -0.23  -0.29  -0.00
                                      -0.41  -4.53   0.47  -0.20
                                      5.75   8.62  -2.24   0.60
                                      -5.14   0.29  -6.00   7.00 ]
                         {:order :row})
                  q (ge factory 4 4 [0.87   0.19   0.45   0.00
                                     0.47  -0.34  -0.77   0.29
                                     0.13   0.32  -0.38  -0.86
                                     0.06  -0.86   0.26  -0.43]
                        {:order :row})
                  ql*c (ge factory 4 4 [73364.78125 54602.0859375 6721.794921875 73364.78125
                                        -23043.171875 -17142.7265625 -2109.498779296875 -23043.171875
                                        6087.68798828125 4543.7509765625 562.00439453125 6087.68798828125
                                        134.61383056640625 98.04141998291016 12.774694442749023 134.61383056640625]
                           {:order :row})]

     (nrm2 (axpy! -1 (gql! a0 (qlf! a0 tau)) q)) => (roughly 0.013)
     (nrm2 (axpy! -1 (doto a1 (qlf! tau)) ql)) => (roughly 0.01123)
     (nrm2 (axpy! -1 (mql! a1 tau c) ql*c)) => (roughly 0.0 0.14))))

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
                               {:order :row})
                  qr (ge factory 6 4 [-17.54  -4.76  -1.96   0.42
                                      -0.52  12.40   7.88  -5.84
                                      -0.40  -0.14  -5.75   4.11
                                      0.44  -0.66  -0.20  -7.78
                                      0.37  -0.26  -0.17  -0.15
                                      -0.29   0.46   0.41   0.24]
                         {:order :row})
                  residual (vctr factory [195.36 107.06])]
     (ls! a b)
     (nrm2 (axpy! -1 residual (ls-residual a b))) => (roughly 0.003 0.0001)
     (nrm2 (axpy! -1 solution (submatrix b 0 0 4 2))) 0.007 => (roughly 0.007)
     (nrm2 (axpy! -1 qr a)) => (roughly 0.0129))))

(defn test-ge-ev [factory]
  (facts
   "LAPACK GE ev!"

   (with-release [a0 (ge factory 5 5 [-1.01,  3.98,  3.30,  4.43,  7.31,
                                      0.86,  0.53,  8.26,  4.96, -6.43,
                                      -4.60, -7.04, -3.89, -7.66, -6.16,
                                      3.31,  5.29,  8.20, -7.33,  2.47,
                                      -4.81,  3.55, -1.51,  6.18,  5.58])
                  a1 (copy a0)
                  eigenvalues (ge factory 5 2 [2.86  10.76
                                               2.86 -10.76
                                               -0.69   4.70
                                               -0.69  -4.70
                                               -10.46   0.00]
                                  {:order :row})
                  vl-res (ge factory 5 5 [-0.04  -0.29  -0.13  -0.33   0.04
                                          -0.62   0.00   0.69   0.00   0.56
                                          0.04   0.58  -0.39  -0.07  -0.13
                                          -0.28  -0.01  -0.02  -0.19  -0.80
                                          0.04  -0.34  -0.40   0.22   0.18]
                             {:order :row})
                  vr-res (ge factory 5 5 [-0.11  -0.17  -0.73   0.00   0.46
                                          -0.41   0.26   0.03   0.02   0.34
                                          -0.10   0.51  -0.19   0.29   0.31
                                          -0.40   0.09   0.08   0.08  -0.74
                                          -0.54   0.00   0.29   0.49   0.16]
                             {:order :row})
                  vl (ge factory 5 5)
                  vr (ge factory 5 5)]

     (nrm2 (axpy! -1 eigenvalues (ev! a0))) => (roughly 0.00943)
     (ev! a1 vl vr) = truthy
     (nrm2 (axpy! -1 (fmap! abs vl-res) (fmap! abs vl))) => (roughly 0.01378)
     (nrm2 (axpy! -1 (fmap! abs vr-res) (fmap! abs vr))) => (roughly 0.010403))))

(defn test-ge-svd [factory]
  (facts
   "LAPACK GE svd!"

   (with-release [a0 (ge factory 6 5 [8.79,  6.11, -9.15,  9.57, -3.49,  9.84,
                                      9.93,  6.91, -7.93,  1.64,  4.02,  0.15,
                                      9.83,  5.04,  4.86,  8.83,  9.80, -8.99,
                                      5.45, -0.27,  4.85,  0.74, 10.00, -6.02,
                                      3.16,  7.98,  3.01,  5.80,  4.27, -5.31])
                  a1 (copy a0)
                  s (vctr factory 5)
                  u (ge factory 6 5)
                  vt (ge factory 5 5)
                  superb (vctr factory 5)
                  s-res (vctr factory [27.47 22.64 8.56 5.99 2.01])
                  u-res (ge factory 6 5 [-0.59   0.26   0.36   0.31   0.23
                                         -0.40   0.24  -0.22  -0.75  -0.36
                                         -0.03  -0.60  -0.45   0.23  -0.31
                                         -0.43   0.24  -0.69   0.33   0.16
                                         -0.47  -0.35   0.39   0.16  -0.52
                                         0.29   0.58  -0.02   0.38  -0.65]
                             {:order :row})
                  vt-res (ge factory 5 5 [-0.25  -0.40  -0.69  -0.37  -0.41
                                          0.81   0.36  -0.25  -0.37  -0.10
                                          -0.26   0.70  -0.22   0.39  -0.49
                                          0.40  -0.45   0.25   0.43  -0.62
                                          -0.22   0.14   0.59  -0.63  -0.44]
                             {:order :row})]

     (nrm2 (axpy! -1 s-res (svd! a0 s superb))) => (roughly 0.007526)
     (nrm2 (axpy! -1 s-res (svd! a1 s u vt superb))) => (roughly 0.007526)
     (nrm2 (axpy! -1 u-res u)) => (roughly 0.01586)
     (nrm2 (axpy! -1 vt-res vt)) => (roughly 0.012377))))

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
  (test-tr factory)
  (test-tr-constructor factory)
  (test-uplo-copy factory tr "TR")
  (test-uplo-swap factory tr "TR")
  (test-uplo-scal factory tr "TR")
  (test-uplo-axpy factory tr "TR")
  (test-tr-mv factory)
  (test-tr-mm factory))

(defn test-blas-host [factory]
  (test-iamin factory)
  (test-imax factory)
  (test-imin factory)
  (test-rot factory)
  (test-rotg factory)
  (test-rotm factory)
  (test-rotmg factory)
  (test-vctr-entry factory)
  (test-vctr-entry! factory)
  (test-vctr-alter! factory)
  (test-vctr-amax factory)
  (test-ge-entry factory)
  (test-ge-entry! factory)
  (test-ge-alter! factory)
  (test-ge-amax factory)
  (test-ge-sum factory)
  (test-ge-dot-strided factory)
  (test-ge-trans! factory)
  (test-tr-entry factory)
  (test-tr-entry! factory)
  (test-tr-bulk-entry! factory)
  (test-tr-alter! factory)
  (test-tr-dot factory)
  (test-tr-nrm2 factory)
  (test-tr-asum factory)
  (test-tr-sum factory)
  (test-tr-amax factory))

(defn test-lapack [factory]
  (test-vctr-srt factory)
  (test-ge-srt factory)
  (test-tr-srt factory)
  (test-ge-trf factory)
  (test-ge-trs factory)
  (test-tr-trs factory)
  (test-ge-det factory)
  (test-ge-sv factory)
  (test-tr-sv factory)
  (test-ge-tri factory)
  (test-tr-tri factory)
  (test-ge-con factory)
  (test-tr-con factory)
  (test-ge-qr factory)
  (test-ge-rq factory)
  (test-ge-lq factory)
  (test-ge-ql factory)
  (test-ge-ls factory)
  (test-ge-ev factory)
  (test-ge-svd factory))

(defn test-blas-sy [factory]
  (test-sy-constructor factory)
  (test-sy factory)
  (test-uplo-copy factory sy "SY")
  (test-uplo-swap factory sy "SY")
  (test-uplo-scal factory sy "SY")
  (test-uplo-axpy factory sy "SY")
  (test-sy-mv factory)
  (test-sy-mm factory))

(defn test-blas-sy-host [factory]
  (test-sy-entry factory)
  (test-sy-entry! factory)
  (test-sy-bulk-entry! factory)
  (test-sy-alter! factory)
  (test-sy-dot factory)
  (test-sy-nrm2 factory)
  (test-sy-asum factory)
  (test-sy-sum factory)
  (test-sy-amax factory))
