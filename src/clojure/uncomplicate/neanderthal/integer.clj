;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.integer
  "Contains type-specific primitive integer functions, equivalents of functions from the
  [[uncomplicate.neanderthal.core]] namespace. Typically, you would want to require this namespace
  if you need to compute real matrices containing longs and/or ints.

  ### Example

      (ns test
        (:require [uncomplicate.neanderthal
                  [core :refer :all :exclude [entry entry! dot nrm2 asum sum]]
                  [integer :refer :all]]))
  "
  (:require [uncomplicate.commons.core :refer [with-release double-fn]]
            [uncomplicate.fluokitten.core :refer [foldmap]]
            [uncomplicate.neanderthal
             [math :refer [sqr]]
             [core :as core]]
            [uncomplicate.neanderthal.internal.api :as api])
  (:import [uncomplicate.neanderthal.internal.api IntegerVector IntegerMatrix IntegerChangeable Vector]))

;; ============ Vector and Matrix access methods ===

(defn entry
  "The primitive, much faster, version of [[uncomplicate.neanderthal.core/entry]]."
  (^long [^IntegerVector x ^long i]
   (if (< -1 i (.dim x))
     (.entry x i)
     (throw (ex-info "Requested element is out of bounds of the vector." {:i i :dim (.dim x)}))))
  (^long [^IntegerMatrix a ^long i ^long j]
   (if (and (< -1 i (.mrows a)) (< -1 j (.ncols a)))
     (.entry a i j)
     (throw (ex-info "Requested element is out of bounds of the matrix."
                     {:i i :j j :mrows (.mrows a) :ncols (.ncols a)})))))

(defn entry!
  "The primitive, much faster, version of [[uncomplicate.neanderthal.core/entry!]]."
  ([^IntegerChangeable x ^long val]
   (.set x val))
  ([^IntegerChangeable x ^long i ^long val]
   (if (< -1 i (.dim x))
     (.set x i val)
     (throw (ex-info "The element you're trying to set is out of bounds of the vector."
                     {:i i :dim (core/dim x)}))))
  ([^IntegerMatrix a ^long i ^long j ^long val]
   (if (and (< -1 i (.mrows a)) (< -1 j (.ncols a)) (.isAllowed ^IntegerChangeable a i j));; TODO isAllowed should check dimensions i and j.
     (.set ^IntegerChangeable a i j val)
     (throw (ex-info "The element you're trying to set is out of bounds of the matrix."
                     {:i i :j j :mrows (.mrows a) :ncols (.ncols a)})))))

(defn dot
  "Primitive wrapper of [[uncomplicate.neanderthal.core/dot]]."
  ^long [x y]
  (core/dot x y))

(defn nrm2
  "Primitive wrapper of [[uncomplicate.neanderthal.core/nrm2]]."
  ^long [x]
  (core/nrm2 x))

(defn asum
  "Primitive wrapper of [[uncomplicate.neanderthal.core/asum]]."
  ^long [x]
  (core/asum x))

(defn sum
  "Primitive wrapper of [[uncomplicate.neanderthal.core/sum]]."
  ^long [x]
  (core/sum x))

(defn amax
  "Primitive wrapper of [[uncomplicate.neanderthal.core/amax]]."
  ^long [x]
  (core/amax x))
