(ns hello-world.core
  (:require [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.real :refer :all]))

;; We create two matrices...
(def a (dge 2 3 [1 2 3 4 5 6]))
(def b (dge 3 2 [1 3 5 7 9 11]))
;; ... and multiply them
(mm a b)

;; If you see something like this:
;; #<DoubleGeneralMatrix| COL, mxn: 2x2, ld:2, ((35.0 44.0) (89.0 116.0))>
;; It means that everything is set up and you can enjoy programming with Neanderthal :)
