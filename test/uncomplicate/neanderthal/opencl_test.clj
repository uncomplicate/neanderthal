(ns uncomplicate.neanderthal.opencl-test
  (:require [midje.sweet :refer :all]
            [vertigo
             [core :refer [wrap]]
             [structs :refer [float64]]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [opencl :refer :all]
             [real :refer [to-buffer sv]]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [max-work-group-size queue-device]]])
  (:import [uncomplicate.neanderthal.protocols
            RealVector RealMatrix]))

(with-default
  (let [cnt 257
        host-x (sv cnt)
        host-y (sv cnt)]
    (facts
     "Carrier methods"
     (with-release [settings (cl-settings *command-queue*)
                    cl-x (cl-fv settings cnt)
                    cl-y (cl-fv settings cnt)]
       (write! cl-x (sv (range cnt)))
       (write! cl-y (sv (range cnt (* 2 cnt))))

       (time (swp cl-x cl-y)) => cl-x

       (read! cl-x host-x)
       (read! cl-y host-y)

       host-x => (sv (range cnt (* 2 cnt)))
       host-y => (sv (range cnt))

       (copy cl-x cl-y) => cl-y

       (read! cl-y host-y)
       host-y =>  host-x


       )

     )))
