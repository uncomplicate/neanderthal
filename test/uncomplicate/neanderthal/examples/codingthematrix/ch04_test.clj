(ns uncomplicate.neanderthal.examples.codingthematrix.ch04-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [real :refer :all]]))

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
   (entry m 1 2) => 30.0

   ))
