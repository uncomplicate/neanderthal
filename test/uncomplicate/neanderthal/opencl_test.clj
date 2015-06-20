(ns uncomplicate.neanderthal.opencl-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.neanderthal
             [protocols :as p]
             [opencl :refer :all]
             [core :refer :all]
             [real :refer :all]]
            [uncomplicate.clojurecl
             [core :refer [with-default with-release *command-queue*]]])
  (:import [uncomplicate.neanderthal.protocols
            RealVector RealMatrix]))

(with-default
  (let [cnt 259
        host-x (sv cnt)
        host-y (sv cnt)
        host-range (sv (range cnt))
        host-range2 (sv (range cnt (* 2 cnt)))]

    (facts
     "RealVector methods"
     (with-release [settings (cl-settings *command-queue*)
                    cl-x (cl-sv settings cnt)]
       (dim cl-x) => cnt

       )
     )

    (facts
     "Carrier methods"
     (with-release [settings (cl-settings *command-queue*)
                    cl-x (cl-sv settings cnt)
                    cl-y (cl-sv settings cnt)]

       (p/byte-size cl-x) => Float/BYTES

       (write! cl-x host-range) => cl-x
       (write! cl-y host-range2) => cl-y

       (read! (zero cl-x) (sv cnt)) => (sv cnt)

       (swp! cl-x cl-y) => cl-x

       (read! cl-x host-x) => host-range2
       (read! cl-y host-y) => host-range

       (copy! cl-x cl-y) => cl-y

       (read! cl-y host-y) => host-x


       )

     )))
