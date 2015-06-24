(ns uncomplicate.neanderthal.opencl-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.neanderthal
             [protocols :as p]
             [opencl :refer :all]
             [core :refer :all]
             [real :refer :all]
             [math :refer [pow]]]
            [uncomplicate.clojurecl
             [core :refer [with-default with-release *command-queue*]]])
  (:import [uncomplicate.neanderthal.protocols
            RealVector RealMatrix]))

(def cnt (long (pow 2 20)))
(def host-z (sv cnt))

(facts
 "RealVector methods"
 (with-default
   (let [host-x (doto (sv cnt) (fset! 3.0))
         host-y (doto (sv cnt) (fset! 2.0))]
     (with-release [settings (cl-settings *command-queue*)
                    cl-x (cl-sv settings cnt)
                    cl-y (cl-sv settings cnt)]

       (dim cl-x) => cnt

       (fset! cl-x 3.0)
       (fset! cl-y 2.0)
       (read! (scal! 2.0 cl-x) (sv cnt)) => (scal! 2.0 host-x)

       (read! (axpy! cl-y 2.0 cl-x) (sv cnt))
       => (axpy! host-y 2.0 host-x)

       (time (dot cl-x cl-y)) => (dot host-x host-y)))))

(with-default
  (facts
   "Carrier methods"
   (let [host-x (doto (sv cnt) (fset! 3.5))
         host-y (doto (sv cnt) (fset! 1.1))]
     (with-release [settings (cl-settings *command-queue*)
                    cl-x (cl-sv settings cnt)
                    cl-y (cl-sv settings cnt)]

       (p/byte-size cl-x) => Float/BYTES

       (write! cl-x host-x) => cl-x
       (write! cl-y host-y) => cl-y

       (read! (zero cl-x) (sv cnt)) => (sv cnt)

       (swp! cl-x cl-y) => cl-x
       (swp! cl-x cl-y) => cl-x

       (read! cl-x (sv cnt)) => host-x

       (copy! cl-x cl-y) => cl-y

       (read! cl-y host-y) => host-x))))
