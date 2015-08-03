(ns uncomplicate.neanderthal.opencl.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.neanderthal
             [native :refer :all]
             [opencl-test :refer :all]]
            [uncomplicate.neanderthal.opencl.amd-gcn
             :refer [gcn-single gcn-double]]))

(test-all gcn-single sge sv)
(test-all gcn-double dge dv)
