;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.block
  (:require [uncomplicate.neanderthal.internal.api
             :refer [dec-property COLUMN_MAJOR ROW_MAJOR UPPER LOWER DIAG_UNIT DIAG_NON_UNIT]])
  (:import [uncomplicate.neanderthal.internal.api Block]))

(defn buffer [^Block x]
  (.buffer x))

(defn offset ^long [^Block x]
  (.offset x))

(defn stride ^long [^Block x]
  (.stride x))
