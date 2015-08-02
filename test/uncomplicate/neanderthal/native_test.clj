(ns uncomplicate.neanderthal.native-test
  (:require [uncomplicate.neanderthal
             [native :refer :all]
             [block :refer [double-accessor float-accessor]]
             [real-test :refer [test-all]]]))

(test-all double-accessor dge dv)
(test-all float-accessor sge sv)
