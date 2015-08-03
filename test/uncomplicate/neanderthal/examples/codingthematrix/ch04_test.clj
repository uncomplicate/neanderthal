(ns uncomplicate.neanderthal.examples.codingthematrix.ch04-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [native :refer [dv dge]]]))

(facts
 "4.1 What is a matrix"
 (let [m (dge 2 3 [1 10 2 20 3 30])]
   (row m 0) => (dv 1 2 3)
   (row m 1) => (dv 10 20 30)
   (col m 0) => (dv 1 10)
   (col m 1) => (dv 2 20)
   (col m 2) => (dv 3 30)
   (mrows m) => 2
   (ncols m) => 3
   (entry m 1 2) => 30.0))

(facts
 "4.4 Transpose"
 (let [m (dge 2 3 [1 2 3 4 5 6])
       mt (trans m)]
   (row m 0) => (col mt 0)
   (trans m) => (dge 3 2 [1 3 5 2 4 6])
   (trans (trans m)) => m))

(facts
 "4.6.5 Algebraic properties of matrix-vector multiplication"
 (let [m (dge 2 3 [1 2 3 4 5 6])
       v (dv 2 3 4)
       u (dv 1 3 5)]
   (mv m (ax 5 v)) => (ax 5 (mv m v))
   (mv m (xpy u v)) => (xpy (mv m u) (mv m v))))
