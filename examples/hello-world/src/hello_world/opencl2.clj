;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns hello-world.opencl2
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl.core :refer [with-default set-default! release-context!]]
            [uncomplicate.neanderthal
             [core :refer [asum]]
             [opencl :refer [clv with-default-engine set-engine!]]]))

;; Conveniently in interactive REPL sessions (but don't do this in production code):

(set-default!)
(set-engine!)

(def gpu-x (clv 1 -2 5))
(asum gpu-x)

(release-context!)

;; Use dynamic bindings or explicit factories in production code:

(with-default
  (with-default-engine
    (with-release [gpu-x (clv 1 -2 5)]
      (asum gpu-x))))
