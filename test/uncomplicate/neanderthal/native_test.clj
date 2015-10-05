(ns uncomplicate.neanderthal.native-test
  (:require [uncomplicate.neanderthal.real-test :refer [test-all]]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]))

(test-all cblas-double)
(test-all cblas-single)
