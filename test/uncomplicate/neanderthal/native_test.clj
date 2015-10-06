(ns uncomplicate.neanderthal.native-test
  (:require [uncomplicate.neanderthal.buffer-block-test :as buffer-block-test]
            [uncomplicate.neanderthal.real-test :as real-test]
            [uncomplicate.neanderthal.impl.cblas
             :refer [cblas-single cblas-double]]))

(buffer-block-test/test-all cblas-double)
(buffer-block-test/test-all cblas-single)
(real-test/test-all cblas-double)
(real-test/test-all cblas-single)
