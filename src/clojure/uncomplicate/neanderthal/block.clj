;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.block
  (:require [uncomplicate.neanderthal.internal.api :as api])
  (:import [uncomplicate.neanderthal.internal.api Block DataAccessor]))

(defn buffer [^Block x]
  (.buffer x))

(defn offset ^long [^Block x]
  (.offset x))

(defn stride ^long [^Block x]
  (.stride x))

(defn wrap-prim [^DataAccessor accessor ^double v]
  (.wrapPrim accessor v))

(defn entry-type [^DataAccessor accessor]
  (.entryType accessor))

(defn entry-width ^long [^DataAccessor accessor]
  (.entryWidth accessor))

(defn count-entries ^long [^DataAccessor accessor data]
  (.count accessor data))

(defn create-data-source [^DataAccessor accessor ^long n]
  (.createDataSource accessor n))

(defn initialize
  ([^DataAccessor accessor buf v]
   (.initialize accessor buf v))
  ([^DataAccessor accessor buf]
   (.initialize accessor buf)))

(defn data-accessor [provider]
  (api/data-accessor provider))
