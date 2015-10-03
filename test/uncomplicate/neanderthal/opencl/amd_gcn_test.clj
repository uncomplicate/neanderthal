(ns uncomplicate.neanderthal.opencl.amd-gcn-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.neanderthal
             [protocols :refer [data-accessor]]
             [native :refer :all]
             [opencl :refer :all]
             [real-test :as rt]
             [opencl-test :refer :all]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]
            [uncomplicate.neanderthal.opencl :refer [gcn-single gcn-double]]))

#_(with-default-engine;;TODO currently throws an exception due to a bug in Clojure 1.8 that soon should be fixed
  (rt/test-all (data-accessor cblas-single) sclge sclv)
  (rt/test-all (data-accessor cblas-double) dclge dclv))
(test-all gcn-single sge sv)
(test-all gcn-double dge dv)
