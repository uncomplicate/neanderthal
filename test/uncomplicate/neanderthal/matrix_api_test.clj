(ns uncomplicate.neanderthal.matrix-api-test
  (:use [clojure test])
  (:use [clojure.core.matrix])
  (:require [uncomplicate.neanderthal.impl.matrix-api]) 
  (:require [clojure.core.matrix.compliance-tester]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(set-current-implementation :neanderthal)

(deftest instance-tests
  (clojure.core.matrix.compliance-tester/instance-test (array :neanderthal [[1 2 3] [4 5 6]]))
  (clojure.core.matrix.compliance-tester/instance-test (array :neanderthal [1 2 3 4 5]))) 

(deftest compliance-test
  (clojure.core.matrix.compliance-tester/compliance-test (array :neanderthal [1 2]))) 