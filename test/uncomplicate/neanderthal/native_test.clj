(ns uncomplicate.neanderthal.native-test
  (:require [uncomplicate.neanderthal
             [native :refer :all]
             [real-test :refer [test-all]]]
            [uncomplicate.neanderthal.impl.buffer-block
             :refer [double-accessor float-accessor]]))

(test-all double-accessor dge dv)
(test-all float-accessor sge sv)
