(ns uncomplicate.neanderthal.opencl-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.neanderthal
             [core :refer :all]
             [math :refer [pow]]
             [opencl :refer :all]]
            [uncomplicate.clojurecl
             [core :refer [with-default with-release *context* *command-queue*]]])
  (:import [uncomplicate.neanderthal.protocols Block]))

(defmacro test-blas1 [engine-factory rv]
  `(facts
    "BLAS methods"
    (with-default
      (let [cnt# (long (+ 1000 (pow 2 12)))
            x-magic# 2
            y-magic# 5
            host-x# (doto (~rv cnt#) (entry! x-magic#))
            host-y# (doto (~rv cnt#) (entry! y-magic#))]
        (with-release [engine# (~engine-factory *context* *command-queue*)
                       cl-x# (clv engine# cnt#)
                       cl-y# (clv engine# cnt#)]

          (dim cl-x#) => cnt#

          (entry! cl-x# x-magic#)
          (entry! cl-y# y-magic#)
          (entry! host-x# 6 -100000.0)
          (write! cl-x# host-x#)

          (float (dot cl-x# cl-y#)) => (float (dot host-x# host-y#))

          (float (asum cl-x#)) => (float (asum host-x#))

          (float (sum cl-x#)) => (float (sum host-x#))

          (nrm2 cl-x#) => (roughly (nrm2 host-x#))

          (iamax cl-x#) => 6

          (read! (scal! 2 cl-x#) (~rv cnt#)) => (scal! 2 host-x#)

          (read! (axpy! 2 cl-x# cl-y#) (~rv cnt#))
          => (axpy! 2 host-x# host-y#)))

      (let [cnt# (long (+ 1000 (pow 2 12)))
            x-magic# 2
            y-magic# 5
            host-x# (doto (~rv cnt#) (entry! 3.5))
            host-y# (doto (~rv cnt#) (entry! 1.1))]
        (with-release [engine# (~engine-factory *context* *command-queue*)
                       cl-x# (clv engine# cnt#)
                       cl-y# (clv engine# cnt#)]

          (write! cl-x# host-x#) => cl-x#
          (write! cl-y# host-y#) => cl-y#

          (with-release [cl-zero# (zero cl-x#)]
            (read! cl-zero# (~rv cnt#))) => (~rv cnt#)

            (swp! cl-x# cl-y#) => cl-x#
            (swp! cl-x# cl-y#) => cl-x#

            (read! cl-x# (~rv cnt#)) => host-x#

            (copy! cl-x# cl-y#) => cl-y#

            (read! cl-y# host-y#) => host-x#)))))

(defmacro test-blas2 [engine-factory rge rv]
  `(facts
    "BLAS 2"
    (with-default
      (let [m-cnt# 2050
            n-cnt# 337
            a-magic# 3
            x-magic# 2
            y-magic# 5
            host-a# (doto (~rge m-cnt# n-cnt#) (entry! a-magic#))
            host-x# (doto (~rv n-cnt#) (entry! x-magic#))
            host-y# (doto (~rv m-cnt#) (entry! y-magic#))]
        (with-release [engine# (~engine-factory *context* *command-queue*)
                       cl-a# (clge engine# m-cnt# n-cnt#)
                       cl-x# (clv engine# n-cnt#)
                       cl-y# (clv engine# m-cnt#)]

          (entry! cl-a# a-magic#)
          (entry! cl-x# x-magic#)
          (entry! cl-y# y-magic#)

          (read! (mv! 10 cl-a# cl-x# 100 cl-y#) (~rv m-cnt#))
          => (mv! 10 host-a# host-x# 100 host-y#))))))

(defn buffer [^Block b]
  (.buffer b))

(defmacro test-blas3 [engine-factory rge rv]
  `(facts
    "BLAS 3"
    (let [m-cnt# 123
          k-cnt# 456
          n-cnt# 789
          host-a# (~rge m-cnt# k-cnt# (repeatedly (* m-cnt# k-cnt#) rand))
          host-b# (~rge k-cnt# n-cnt# (map (partial * 2) (repeatedly (* k-cnt# n-cnt#) rand)))
          host-c# (~rge m-cnt# n-cnt# (map (partial * 2) (repeatedly (* m-cnt# n-cnt#) rand)))]
      (with-default
        (with-release [engine# (~engine-factory *context* *command-queue*)
                       cl-a# (clge engine# m-cnt# k-cnt#)
                       cl-b# (clge engine# k-cnt# n-cnt#)
                       cl-c# (clge engine# m-cnt# n-cnt#)]

          (write! cl-a# host-a#)
          (write! cl-b# host-b#)
          (write! cl-c# host-c#)

          (< (double (nrm2 (~rv (buffer (axpy! -1 (mm! 10 host-a# host-b# 100 host-c#)
                                               (read! (mm! 10 cl-a# cl-b# 100 cl-c#)
                                                      (~rge m-cnt# n-cnt#)))))))
             (* 2 m-cnt# n-cnt# k-cnt# 1e-8)) => true)))))

(defmacro test-all [engine-factory rge rv]
  `(do
     (test-blas1 ~engine-factory ~rv)
     (test-blas2 ~engine-factory ~rge ~rv)
     (test-blas3 ~engine-factory ~rge ~rv)))
