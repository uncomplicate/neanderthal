;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.block
  (:require [uncomplicate.neanderthal.protocols :refer [dec-property]])
  (:import [uncomplicate.neanderthal.protocols Block ContiguousBlock]))

(defn buffer [^Block x]
  (.buffer x))

(defn offset ^long [^Block x]
  (.offset x))

(defn stride ^long [^Block x]
  (.stride x))

(defn order ^long [^ContiguousBlock x]
  (dec-property (.order x)))

(defn block? [x]
  (instance? Block x))

(defn ecount
  "Returns the total number of elements in all dimensions of a block x
  of (possibly strided) memory.

  (ecount (dv 1 2 3)) => 3
  (ecount (dge 2 3)) => 6
  "
  ^long [^Block x]
  (.count x))
