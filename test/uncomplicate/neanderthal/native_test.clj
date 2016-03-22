(ns uncomplicate.neanderthal.native-test
  (:require [uncomplicate.neanderthal.block-test :as block-test]
            [uncomplicate.neanderthal.real-test :as real-test]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]))

(block-test/test-all cblas-double)
(block-test/test-all cblas-single)
(real-test/test-all cblas-double)
(real-test/test-all cblas-single)
