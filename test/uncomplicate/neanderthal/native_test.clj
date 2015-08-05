(ns uncomplicate.neanderthal.native-test
  (:require [uncomplicate.neanderthal
             [protocols :refer [data-accessor]]
             [native :refer :all]
             [real-test :refer [test-all]]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]))

(test-all (data-accessor cblas-double) dge dv)
(test-all (data-accessor cblas-single) sge sv)
