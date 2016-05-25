(ns hello-world.opencl1
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl
             [core :refer [with-platform platforms with-context context
                           with-queue sort-by-cl-version devices ]]
             [legacy :refer [with-default-1 command-queue-1]]]
            [uncomplicate.neanderthal
             [core :refer [asum]]
             [opencl :refer [clv with-default-engine]]]))

;; Important note: this will use the default device, whatever it is
;; on your machine. This device might be old and not very capable,
;; especially if it's a CPU on your not-very-recent Mac.

(with-default-1
  (with-default-engine
    (with-release [gpu-x (clv 1 -2 5)]
      (asum gpu-x))))

;; In that case, try something like this:
(with-platform (first (platforms))
  (let [dev (first (sort-by-cl-version (devices :gpu)))]
    (with-context (context [dev])
      (with-queue (command-queue-1 dev)
        (with-default-engine
          (with-release [gpu-x (clv 1 -2 5)]
            (asum gpu-x)))))))
