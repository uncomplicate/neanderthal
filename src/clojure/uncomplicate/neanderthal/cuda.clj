;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.cuda
  "Specialized constructors that use CUDA engine by default, and convenient macros for
  creating and maintaining engines in appropriate CUDA and cuBLAS context.
  A convenience over agnostic [[uncomplicate.neanderthal.core]] functions."
  (:require [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.neanderthal.core :refer [vctr ge tr]]
            [uncomplicate.neanderthal.internal.device.cublas
             :refer [cublas-double cublas-float cublas-handle]])
  (:import [uncomplicate.neanderthal.internal.api Block DataAccessor]))

(def ^{:dynamic true
       :doc "Dynamically bound CUDA factory that is used in vector and matrix constructors."}
  *cuda-factory*)

(def ^{:doc "Constructor of a single precision floating point CUDA factory."}
  cuda-float cublas-float)

(def ^{:doc "Constructor of a double-precision floating point CUDA factory."}
  cuda-double cublas-double)

(def  ^{:doc "Creates the context handler for the cuda BLAS engine."}
  cuda-handle cublas-handle)

(defmacro with-engine
  "Creates a CUDA factory using the provided `factory` constructor function and a `handle` cuBLAS
  context handler. The created factory will work using the provided cuBLAS `handle` and its context,
  and will be bound to [[*cuda-factory*]].
  Enables the use of [[cuv]], [[cuge]], [[cutr]], etc. in its body.
  "
  [factory handle & body]
  `(binding [*cuda-factory* (~factory ~handle)]
     (try
       ~@body
       (finally (release *cuda-factory*)))))

(defmacro with-default-engine
  "Creates a CUDA factory using the default factory and a new implicit cuBLAS context handler.
  Enables the use of [[cuv]], [[cuge]], [[cutr]], etc. in its body.
  "
  [& body]
  `(with-release [handle# (cuda-handle)]
     (with-engine cuda-float handle# ~@body)))

(defn cuv
  "Creates a vector using CUDA GPU engine provided to the bound [[*cuda-factory*]]
  (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr *cuda-factory* source))
  ([x & xs]
   (cuv (cons x xs))))

(defn cuge
  "Creates a GE matrix using CUDA GPU engine provided to the bound [[*cuda-factory*]]
  (see [[uncomplicate.neanderthal.core/ge]])."
  ([^long m ^long n source options]
   (ge *cuda-factory* m n source options))
  ([^long m ^long n arg]
   (ge *cuda-factory* m n arg))
  ([^long m ^long n]
   (ge *cuda-factory* m n))
  ([a]
   (ge *cuda-factory* a)))

(defn cutr
  "Creates a TR matrix using CUDA GPU engine provided to the bound [[*cuda-factory*]]
  (see [[uncomplicate.neanderthal.core/tr]])."
  ([^long n source options]
   (tr *cuda-factory* n source options))
  ([^long n arg]
   (tr *cuda-factory* n arg))
  ([arg]
   (tr *cuda-factory* arg)))
