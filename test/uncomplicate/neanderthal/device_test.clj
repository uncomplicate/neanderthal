;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.device-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [math :refer [pow]]
             [vect-math :refer [linear-frac!]]
             [block :refer [buffer]]]
            [uncomplicate.neanderthal.internal.api :refer [native-factory data-accessor]]))

(defn test-clblock [factory]
  (let [host-factory (native-factory factory)
        m 33
        n (long (+ 1000 (pow 2 12)))
        cnt n
        x-magic 2
        y-magic 5
        magic 17.0]
    (with-release [host-a (entry! (ge host-factory m n) x-magic)
                   host-b (entry! (ge host-factory m n) y-magic)
                   host-x (entry! (vctr host-factory cnt) x-magic)
                   host-y (entry! (vctr host-factory cnt) y-magic)
                   cl-a (ge factory m n host-a)
                   cl-b (ge factory m n)
                   cl-x (vctr factory cnt)
                   cl-y (vctr factory cnt)
                   cl-z (vctr factory cnt)
                   row-a (row cl-a 3)
                   subvector-x (subvector cl-x 3 (/ cnt 2))]
      (facts
       "Device Vector: equality and hashCode."
       (= cl-x nil) => false
       cl-x => cl-x
       (transfer! host-x cl-x) => (transfer! host-x cl-y)
       (= cl-x cl-z) => false
       (transfer! host-y cl-y) =not=> cl-x
       row-a => cl-x
       (row cl-a 4) =not=> (col cl-a 4))

      (facts
       "Device Matrix: equality and hashCode."
       cl-a => cl-a
       (transfer! host-a cl-b) => cl-a
       (transfer! host-b cl-b) =not=> cl-a
       cl-a =not=> cl-x)

      (facts
       "Matrix rows and columns."
       (dim subvector-x) => (/ cnt 2)
       (transfer! (entry! host-x 3 magic) cl-x) => cl-x
       (entry (transfer subvector-x) 0)
       => magic
       (entry (transfer (copy! (subvector cl-x 3 (mrows cl-a))
                               (copy (col cl-a 6)))) 0)
       => magic))))

(defn test-blas1 [factory]
  (facts
   "BLAS methods"
   (let [host-factory (native-factory factory)
         cnt (long (+ 1000 (pow 2 12)))
         x-magic 2
         y-magic 5]
     (with-release [host-x (entry! (vctr host-factory cnt) x-magic)
                    host-y (entry! (vctr host-factory cnt) y-magic)
                    cl-x (vctr factory cnt)
                    cl-y (vctr factory cnt)
                    cl-a (doto (ge factory 1000 911) (entry! x-magic))]

       (entry! host-x 6 -100000.0)
       (transfer! host-x cl-x)
       (transfer! host-y cl-y)

       (float (dot cl-x cl-y)) => (float (dot host-x host-y))

       (float (asum cl-x)) => (float (asum host-x))

       (float (sum cl-x)) => (float (sum host-x))

       (nrm2 cl-x) => (roughly (nrm2 host-x))

       (iamax cl-x) => 6

       (transfer (scal! 2 cl-x)) => (scal! 2 host-x)

       (transfer (axpy! 2 cl-x cl-y)) => (axpy! 2 host-x host-y)

       (sum cl-a) => (double (* x-magic 1000 911 ))))

   (let [host-factory (native-factory factory)
         cnt (long (+ 1000 (pow 2 12)))
         x-magic 2
         y-magic 5]
     (with-release [host-x (entry! (vctr host-factory cnt) 3.5)
                    host-y (entry! (vctr host-factory cnt) 1.1)
                    cl-x (vctr factory cnt)
                    cl-y (vctr factory cnt)]

       (transfer! host-x cl-x) => cl-x
       (transfer! host-y cl-y) => cl-y

       (with-release [cl-zero (zero cl-x)]
         (transfer cl-zero)) => (vctr host-factory cnt)

       (swp! cl-x cl-y) => cl-x
       (swp! cl-x cl-y) => cl-x

       (transfer cl-x) => host-x

       (copy! cl-x cl-y) => cl-y

       (transfer! cl-y host-y) => host-x))))

(defn test-blas2 [factory]
  (facts
   "BLAS 2"
   (let [host-factory (native-factory factory)
         m-cnt 2050
         n-cnt 337
         a-magic 3
         x-magic 2
         y-magic 5]
     (with-release [host-a (entry! (ge host-factory m-cnt n-cnt) a-magic)
                    host-x (entry! (vctr host-factory n-cnt) x-magic)
                    host-y (entry! (vctr host-factory m-cnt) y-magic)
                    cl-a-master (ge factory (* 3 m-cnt) (* 3 n-cnt))
                    cl-a (transfer! host-a (submatrix cl-a-master m-cnt n-cnt m-cnt n-cnt))
                    cl-x-master (ge factory m-cnt n-cnt)
                    cl-y-master (ge factory n-cnt m-cnt)
                    cl-x (transfer! host-x (row cl-x-master 1))
                    cl-y (transfer! host-y (row cl-y-master 1))]

       (native (mv! 10 cl-a cl-x 100 cl-y)) => (mv! 10 host-a host-x 100 host-y)))))

(defn test-blas3 [factory]
  (facts
   "BLAS 3"
   (let [host-factory (native-factory factory)
         m-cnt 123
         k-cnt 456
         n-cnt 789]
     (with-release [host-a (ge host-factory m-cnt k-cnt
                               (repeatedly (* m-cnt k-cnt) rand))
                    host-b (ge host-factory k-cnt n-cnt
                               (map (partial * 2)
                                    (repeatedly (* k-cnt n-cnt) rand)))
                    host-c (ge host-factory m-cnt n-cnt
                               (map (partial * 2)
                                    (repeatedly (* m-cnt n-cnt) rand)))
                    cl-a-master (ge factory (* 2 m-cnt) (* 2 k-cnt))
                    cl-b-master (ge factory (* 2 k-cnt) (* 2 n-cnt))
                    cl-c-master (ge factory (* 2 m-cnt) (* 2 n-cnt))
                    cl-a (transfer! host-a (submatrix cl-a-master m-cnt k-cnt m-cnt k-cnt))
                    cl-b (transfer! host-b (submatrix cl-b-master k-cnt n-cnt k-cnt n-cnt))
                    cl-c (transfer! host-c (submatrix cl-c-master m-cnt n-cnt m-cnt n-cnt))]

       (let [diff (axpy! -1 (mm! 10 host-a host-b 100 host-c)
                         (transfer host-factory (mm! 10 cl-a cl-b 100 cl-c)))
             diff-vec (vctr host-factory (* m-cnt n-cnt))]
         (dotimes [j m-cnt]
           (copy! (col diff j) (subvector diff-vec (* j m-cnt) m-cnt)))
         (< (double (nrm2 diff-vec)) (* 2 m-cnt n-cnt k-cnt 1e-8))) => true))))

(defn test-vect-math [factory]
  (facts
   "vect-math arrary overflow test"
   (let [host-factory (native-factory factory)
         cnt 10000
         x-magic 2]
     (with-release [host-x (entry! (vctr host-factory cnt) x-magic)
                    cl-x (vctr factory cnt)]

       (transfer! host-x cl-x)
       (linear-frac! (subvector cl-x 3000 4000) 100)
       (linear-frac! (subvector host-x 3000 4000) 100)
       (transfer cl-x) => host-x))))

(defn test-all [factory]
  (do
    (test-clblock factory)
    (test-blas1 factory)
    (test-blas2 factory)
    (test-blas3 factory)
    (test-vect-math factory)))
