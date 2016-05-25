(ns hello-world.opencl1
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl.legacy :refer [with-default-1]]
            [uncomplicate.neanderthal
             [core :refer [asum]]
             [opencl :refer [clv with-default-engine]]]))

(with-default-1
  (with-default-engine
    (with-release [gpu-x (clv 1 -2 5)]
      (asum gpu-x))))
