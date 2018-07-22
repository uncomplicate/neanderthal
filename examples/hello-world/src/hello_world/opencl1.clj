;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns hello-world.opencl1
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl
             [core :refer [with-platform platforms with-context context with-queue
                           sort-by-cl-version devices with-default-1 command-queue-1
                           set-default-1! release-context!]]]
            [uncomplicate.neanderthal
             [core :refer [asum]]
             [opencl :refer [clv with-default-engine set-engine!]]]))

;; Conveniently in interactive REPL sessions (but don't do this in production code):

(set-default-1!)
(set-engine!)

(def gpu-x (clv 1 -2 5))
(asum gpu-x)

(release-context!)

;; Use dynamic bindings or explicit factories in production code:

(with-default-1
  (with-default-engine
    (with-release [gpu-x (clv 1 -2 5)]
      (asum gpu-x))))

;; Important note: this will use the default device, whatever it is
;; on your machine. This device might be old and not very capable,
;; especially if it's a CPU on your not-very-recent Mac.

;; In that case, try something like this:
(with-platform (first (platforms))
  (let [dev (first (sort-by-cl-version (devices :gpu)))]
    (with-context (context [dev])
      (with-queue (command-queue-1 dev)
        (with-default-engine
          (with-release [gpu-x (clv 1 -2 5)]
            (asum gpu-x)))))))
