(ns uncomplicate.neanderthal.opencl-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [protocols :refer [factory data-accessor]]
             [core :refer :all]
             [opencl :refer :all]
             [math :refer [pow]]])
  (:import [uncomplicate.neanderthal.protocols Block]))

(defn test-clblock [ocl-factory]
  (let [host-factory (factory (data-accessor ocl-factory))
        m 33
        n (long (+ 1000 (pow 2 12)))
        cnt n
        x-magic 2
        y-magic 5
        magic 17.0]
    (with-release [host-a (entry! (create-ge-matrix host-factory m n) x-magic)
                   host-b (entry! (create-ge-matrix host-factory m n) y-magic)
                   host-x (entry! (create-vector host-factory cnt) x-magic)
                   host-y (entry! (create-vector host-factory cnt) y-magic)
                   cl-a (create-ge-matrix ocl-factory m n host-a)
                   cl-b (create-ge-matrix ocl-factory m n)
                   cl-x (create-vector ocl-factory cnt)
                   cl-y (create-vector ocl-factory cnt)
                   cl-z (create-vector ocl-factory cnt)
                   row-a (row cl-a 3)
                   subvector-x (subvector cl-x 3 (/ cnt 2))]
      (facts
       "OpenCL Vector: equality and hashCode."
       (= cl-x nil) => false
       cl-x => cl-x
       (transfer! host-x cl-x) => (transfer! host-x cl-y)
       (= cl-x cl-z) => false
       (transfer! host-y cl-y) =not=> cl-x
       row-a => cl-x
       (row cl-a 4) =not=> (col cl-a 4))

      (facts
       "OpenCL Matrix: equality and hashCode."
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

(defn test-blas1 [ocl-factory]
  (facts
   "BLAS methods"
   (let [host-factory (factory (data-accessor ocl-factory))
         cnt (long (+ 1000 (pow 2 12)))
         x-magic 2
         y-magic 5]
     (with-release [host-x (entry! (create-vector host-factory cnt) x-magic)
                    host-y (entry! (create-vector host-factory cnt) y-magic)
                    cl-x (create-vector ocl-factory cnt)
                    cl-y (create-vector ocl-factory cnt)]

       (dim cl-x) => cnt

       (entry! host-x 6 -100000.0)
       (transfer! host-x cl-x)
       (transfer! host-y cl-y)

       (float (dot cl-x cl-y)) => (float (dot host-x host-y))

       (float (asum cl-x)) => (float (asum host-x))

       (float (sum cl-x)) => (float (sum host-x))

       (nrm2 cl-x) => (roughly (nrm2 host-x))

       (iamax cl-x) => 6

       (transfer (scal! 2 cl-x)) => (scal! 2 host-x)

       (transfer (axpy! 2 cl-x cl-y)) => (axpy! 2 host-x host-y)))

   (let [host-factory (factory (data-accessor ocl-factory))
         cnt (long (+ 1000 (pow 2 12)))
         x-magic 2
         y-magic 5]
     (with-release [host-x (entry! (create-vector host-factory cnt) 3.5)
                    host-y (entry! (create-vector host-factory cnt) 1.1)
                    cl-x (create-vector ocl-factory cnt)
                    cl-y (create-vector ocl-factory cnt)]

       (transfer! host-x cl-x) => cl-x
       (transfer! host-y cl-y) => cl-y

       (with-release [cl-zero (zero cl-x)]
         (transfer cl-zero)) => (create-vector host-factory cnt)

       (swp! cl-x cl-y) => cl-x
       (swp! cl-x cl-y) => cl-x

       (transfer cl-x) => host-x

       (copy! cl-x cl-y) => cl-y

       (transfer! cl-y host-y) => host-x))))

(defn test-blas2 [ocl-factory]
  (facts
   "BLAS 2"
   (let [host-factory (factory (data-accessor ocl-factory))
         m-cnt 2050
         n-cnt 337
         a-magic 3
         x-magic 2
         y-magic 5]
     (with-release [host-a (entry! (create-ge-matrix host-factory m-cnt n-cnt) a-magic)
                    host-x (entry! (create-vector host-factory n-cnt) x-magic)
                    host-y (entry! (create-vector host-factory m-cnt) y-magic)
                    cl-a-master (create-ge-matrix ocl-factory (* 3 m-cnt) (* 3 n-cnt))
                    cl-a (transfer! host-a (submatrix cl-a-master m-cnt n-cnt m-cnt n-cnt))
                    cl-x-master (create-ge-matrix ocl-factory m-cnt n-cnt)
                    cl-y-master (create-ge-matrix ocl-factory n-cnt m-cnt)
                    cl-x (transfer! host-x (row cl-x-master 1))
                    cl-y (transfer! host-y (row cl-y-master 1))]
       (transfer! (mv! 10 cl-a cl-x 100 cl-y)
                  (create-vector host-factory m-cnt))
       => (mv! 10 host-a host-x 100 host-y)))))

(defn buffer [^Block b]
  (.buffer b))

(defn test-blas3 [ocl-factory]
  (facts
   "BLAS 3"
   (let [host-factory (factory (data-accessor ocl-factory))
         m-cnt 123
         k-cnt 456
         n-cnt 789]
     (with-release [host-a (create-ge-matrix host-factory m-cnt k-cnt
                                             (repeatedly (* m-cnt k-cnt) rand))
                    host-b (create-ge-matrix host-factory k-cnt n-cnt
                                             (map (partial * 2)
                                                  (repeatedly (* k-cnt n-cnt) rand)))
                    host-c (create-ge-matrix host-factory m-cnt n-cnt
                                             (map (partial * 2)
                                                  (repeatedly (* m-cnt n-cnt) rand)))
                    cl-a-master (create-ge-matrix ocl-factory (* 2 m-cnt) (* 2 k-cnt))
                    cl-b-master (create-ge-matrix ocl-factory (* 2 k-cnt) (* 2 n-cnt))
                    cl-c-master (create-ge-matrix ocl-factory (* 2 m-cnt) (* 2 n-cnt))
                    cl-a (transfer! host-a (submatrix cl-a-master m-cnt k-cnt m-cnt k-cnt))
                    cl-b (transfer! host-b (submatrix cl-b-master k-cnt n-cnt k-cnt n-cnt))
                    cl-c (transfer! host-c (submatrix cl-c-master m-cnt n-cnt m-cnt n-cnt))]

       (< (double
           (nrm2 (create-vector
                  host-factory
                  (buffer (axpy! -1 (mm! 10 host-a host-b 100 host-c)
                                 (create-ge-matrix host-factory m-cnt n-cnt
                                                   (mm! 10 cl-a cl-b 100 cl-c)))))))
          (* 2 m-cnt n-cnt k-cnt 1e-8))
       => true))))

(defn test-all [ocl-factory]
  (do
    (test-clblock ocl-factory)
    (test-blas1 ocl-factory)
    (test-blas2 ocl-factory)
    (test-blas3 ocl-factory)))
