;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.random
  "TODO
  "
  (:require [uncomplicate.commons.utils :refer [generate-seed]]
            [uncomplicate.neanderthal.internal.api :as api]))

(defn rng-state
  "Creates a random number generator state compatible with a factory.
  Optionally accepts the seed for reproducibility."
  ([fact ^long seed]
   (api/create-rng-state (api/factory fact) seed))
  ([fact]
   (rng-state fact (generate-seed))))

(defn rand-normal!
  "Populates vector `x` with normally distributed random numbers.
  The distribution is controlled by arguments `mu` and `sigma`.
  If `rng-state` is not provided, uses the global default rng state."
  ([x]
   (api/rand-normal (api/engine x) nil 0 1 x))
  ([mu sigma x]
   (api/rand-normal (api/engine x) nil mu sigma x))
  ([rng-state x]
   (api/rand-normal (api/engine x) rng-state 0 1 x))
  ([rng-state mu sigma x]
   (api/rand-normal (api/engine x) rng-state mu sigma x)))

(defn rand-uniform!
  "Populates vector `x` with uniformly distributed random numbers.
  The distribution is controlled by arguments `lower` and `upper`.
  If `rng-state` is not provided, uses the engine's global default rng state."
  ([x]
   (api/rand-uniform (api/engine x) nil 0 1 x))
  ([lower upper x]
   (api/rand-uniform (api/engine x) nil lower upper x))
  ([rng-state x]
   (api/rand-uniform (api/engine x) rng-state 0 1 x))
  ([rng-state lower upper x]
   (api/rand-uniform (api/engine x) rng-state lower upper x)))
