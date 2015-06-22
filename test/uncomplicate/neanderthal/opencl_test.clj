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

(with-default
  (let [cnt (long (+ (pow 2 28)))
        ;;host-x (sv cnt)
        ;;host-y (sv cnt)
        ;;host-range (sv (range cnt))
        ;;host-range2 (sv (range cnt (* 2 cnt)))
        ]

    (facts
     "RealVector methods"
     (with-release [settings (cl-settings *command-queue*)
                    cl-x (cl-sv settings cnt)
                    cl-y (cl-sv settings cnt)]

       (comment (dim cl-x) => cnt

                (fset! cl-x 2.5)
                (read! (scal! 1.5 cl-x) (sv cnt)) => (sv (take cnt (repeat 3.75)))

                (write! cl-x host-range)
                (write! cl-y host-range2)

                (read! (axpy! cl-y 1.5 cl-x) (sv cnt))
                => (axpy! host-range2 1.5 host-range))

       (fset! cl-x 2.0)
       (fset! cl-y 5.0)
       (time (dot cl-x cl-y)) => (float (* cnt 10))

       )
     )

    (comment (facts
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

              ))))
