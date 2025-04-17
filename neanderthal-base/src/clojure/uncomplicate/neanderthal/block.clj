;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.block
  "Convenient functions for accessing the memory block that holds vector space's data
  and inquire about its structure. This is less useful for application code, which should
  use linear algebra functions instead of snooping inside, but is indispensible in code that
  extends Neanderthal's functionality.
  "
  (:require [uncomplicate.neanderthal.internal.api :as api]
            [uncomplicate.fluokitten.protocols :refer [extract]])
  (:import [uncomplicate.neanderthal.internal.api Block DataAccessor LayoutNavigator DenseStorage
            FullStorage]))

(defn buffer
  "Accesses the underlying data buffer. Typically returns the JavaCPP pointer to a memory block,
  or a technology-specific wrapper that serves an equivalent function. You might need to further
  call `extract` on it. See the implementation of Neanderthal's backends for real-world examples.
  This is trivial on the surface, but many traps wait in disguise; please study Clojure CPP
  to understand how this works.
  "
  [^Block x]
  (.buffer x))

(defn offset
  "Returns offset in the underlying memory block that `x` actually controls. Since the transition
  to JavaCPP, this usually returns `0`, and the pointer itself is responsible for handling offsets.
  Still used in OpenCL backend (based on JOCL), but a long term goal is to replace this with JavaCPP
  and deprecate this function.
  "
  ^long [^Block x]
  (.offset x))

(defn stride
  "Returns stride between two adjacent group of elements in the underlying memory that holds `x` 's data."
  ^long [^Block x]
  (.stride x))

(defn column?
  "Checks whether matrix `a` is laid column by column."
  [a]
  (.isColumnMajor (api/navigator a)))

(defn row?
  "Checks whether matrix `a` is laid row by row."
  [a]
  (.isRowMajor (api/navigator a)))

(defn contiguous?
  "Checks whether vector space `x` is 'dense', without gaps and strides."
  [^Block x]
  (.isContiguous x))

(defn cast-prim
  "Casts value `v` into the appropriate primitive number determined by the data `accessor`.
  Mostly useful for satisfying the compiler complaints in macros."
  [^DataAccessor accessor v]
  (.castPrim accessor v))

(defn entry-type
  "Returns the type of entries that this data accessor manages (double, float, etc.)."
  [^DataAccessor accessor]
  (.entryType accessor))

(defn entry-width
  "Returns the width in bytes of entries that this data accessor manages (8 for double, etc.)."
  ^long [^DataAccessor accessor]
  (.entryWidth accessor))

(defn count-entries
  "Counts the number of entries in `data` according to the data accessor's way of counting."
  ^long [^DataAccessor accessor data]
  (.count accessor data))

(defn create-data-source
  "Creates a memory block that can hold `n` entries according to data accessor's way of counting."
  [da-provider ^long n]
  (.createDataSource (api/data-accessor da-provider) n))

(defn initialize!
  "Initializes the buffer with the provied value, or zero."
  ([da-provider buf v]
   (.initialize (api/data-accessor da-provider) (extract buf) v))
  ([da-provider buf]
   (.initialize (api/data-accessor da-provider) (extract buf))))

(defn data-accessor
  "Returns `provider` 's data accessor."
  [provider]
  (api/data-accessor provider))
