(ns hello-world.opencl2
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl.core :refer [with-default]]
            [uncomplicate.neanderthal
             [core :refer [asum]]
             [opencl :refer [clv with-default-engine]]]))

(with-default
  (with-default-engine
    (with-release [gpu-x (clv 1 -2 5)]
      (asum gpu-x))))
