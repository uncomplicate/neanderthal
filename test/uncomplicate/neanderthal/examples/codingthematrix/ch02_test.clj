(ns uncomplicate.neanderthal.examples.codingthematrix.ch02-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [double :refer :all]]))

(facts
 "2.1 What is a Vector?"
 (let [v (dv 3.14159 2.718281828 -1.0 2.0)]        
   (dim v) => 4
   (vect? v ) => true
   
   (ifn? v) => true
   (v 0) => (entry v 0)
   (v 1) => (entry v 1)
   (v 2) => (entry v 2)
   (v 3) => (entry v 3)))

;; Some sections, such as 2.2 and 2.2.1 are not
;; applicable for this library, since they are
;; not numeric in nature.
;; Many examples from the book directly apply to
;; Clojure maps, which makes sense, but is not of
;; interest for numeric computations.
;; Such discussions have been skipped here.

  
(facts
 "2.4 Vector addition"

 (let [trans-vector (dv [1 2])
       translate (fn [x] (axpy! x trans-vector))]

   (translate (dv 4 4)) => (dv 5 6)
   (translate (dv -4 -4)) => (dv -3 -2)

   (dv 2) => (dv 0 0)

   (axpy! (dv 4 4) (dv 2)) => (dv 4 4)
   (axpy! (dv 4 4) (zero (dv 4 4))) => (dv 4 4)

   
   ))
        
