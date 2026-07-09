;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.cuda
  "Contains specialized constructors that use CUDA engine by default, and convenient macros for
  creating and maintaining engines in appropriate CUDA and cuBLAS context. A convenience over
  agnostic [[uncomplicate.neanderthal.core]] functions."
  (:require [uncomplicate.commons
             [core :refer [release with-release]]
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.clojure-cpp :refer [null?]]
            [uncomplicate.clojurecuda.core
             :refer [current-context default-stream push-context! context]]
            [uncomplicate.neanderthal
             [core :refer [vctr ge tr sy]]
             [native :refer [native-float native-double native-half native-long native-int
                             native-short native-byte]]]
            [uncomplicate.neanderthal.internal.cpp.cuda.factory
             :refer [cublas-double cublas-float cublas-half cublas-long cublas-int cublas-short cublas-byte]]))

(def ^{:dynamic true
       :doc "Dynamically bound CUDA factory that is used in vector and matrix constructors."}
  *cuda-factory*)

(defn cuda-float
  "Constructor of a single precision floating point CUDA factory."
  ([ctx strm]
   (cublas-float native-float ctx strm))
  ([strm]
   (cuda-float (current-context) strm))
  ([]
   (let [ctx (current-context)]
     (cuda-float (if (null? ctx) (push-context! (context)) ctx) default-stream))))

(defn cuda-double
  "Constructor of a double-precision floating point CUDA factory."
  ([ctx strm]
   (cublas-double native-double ctx strm))
  ([strm]
   (cuda-double (current-context) strm))
  ([]
   (let [ctx (current-context)]
     (cuda-double (if (null? ctx) (push-context! (context)) ctx) default-stream))))

(defn cuda-half
  "Constructor of a half-precision floating point CUDA factory."
  ([ctx strm]
   (cublas-half native-half ctx strm))
  ([strm]
   (cuda-half (current-context) strm))
  ([]
   (let [ctx (current-context)]
     (cuda-half (if (null? ctx) (push-context! (context)) ctx) default-stream))))

(defn cuda-long
  "Constructor of a long CUDA factory."
  ([ctx strm]
   (cublas-long native-long ctx strm))
  ([strm]
   (cuda-long(current-context) strm))
  ([]
   (let [ctx (current-context)]
     (cuda-long (if (null? ctx) (push-context! (context)) ctx) default-stream))))

(defn cuda-int
  "Constructor of an int CUDA factory."
  ([ctx strm]
   (cublas-int native-int ctx strm))
  ([strm]
   (cuda-int (current-context) strm))
  ([]
   (let [ctx (current-context)]
     (cuda-int (if (null? ctx) (push-context! (context)) ctx) default-stream))))

(defn cuda-short
  "Constructor of a short CUDA factory."
  ([ctx strm]
   (cublas-short native-short ctx strm))
  ([strm]
   (cuda-short (current-context) strm))
  ([]
   (let [ctx (current-context)]
     (cuda-short (if (null? ctx) (push-context! (context)) ctx) default-stream))))

(defn cuda-byte
  "Constructor of a byte CUDA factory."
  ([ctx strm]
   (cublas-byte native-byte ctx strm))
  ([strm]
   (cuda-byte (current-context) strm))
  ([]
   (let [ctx (current-context)]
     (cuda-byte (if (null? ctx) (push-context! (context)) ctx) default-stream))))

(defn factory-by-type [data-type]
  (case data-type
    :float cuda-float
    :double cuda-double
    :half cuda-half
    :long cuda-long
    :int cuda-int
    :short cuda-short
    :byte cuda-byte
    :uint8 cuda-byte
    (cond
      (= Float/TYPE data-type) cuda-float
      (= float data-type) cuda-float
      (= Double/TYPE data-type) cuda-double
      (= double data-type) cuda-double
      (= int data-type) cuda-int
      (= Integer/TYPE data-type) cuda-int
      (= long data-type) cuda-long
      (= Long/TYPE data-type) cuda-long
      (= short data-type) cuda-short
      (= Short/TYPE data-type) cuda-short
      (= byte data-type) cuda-byte
      (= Byte/TYPE data-type) cuda-byte
      :default (dragan-says-ex "You requested a factory for an unsupported data type."
                               {:requested data-type :available [:float :double Float/TYPE Double/TYPE]}))))

(defn set-engine!
  "Creates an CUDA factory using the provided `factory` constructor function. The created factory
  will work using the provided stream and the current context, and will be bound to the root of
  [[*cuda-factory*]]. Enables the use of [[cuv]], [[cuge]], [[cutr]], etc. globally.
  "
  ([factory hstream]
   (alter-var-root (var *cuda-factory*) (constantly (factory (current-context) hstream))))
  ([factory]
   (set-engine! factory default-stream))
  ([]
   (set-engine! cuda-float default-stream)))

(defmacro with-engine
  "Creates a CUDA factory using the provided `factory` constructor function and a `handle` cuBLAS
  context handler. The created factory will work using the provided cuBLAS `handle` and its context,
  and will be bound to [[*cuda-factory*]].
  Enables the use of [[cuv]], [[cuge]], [[cutr]], etc. in its body.
  "
  [factory hstream & body]
  `(binding [*cuda-factory* (~factory (current-context) ~hstream)]
     (try
       ~@body
       (finally (release *cuda-factory*)))))

(defmacro with-default-engine
  "Creates a CUDA factory using the default factory and a new implicit cuBLAS context handler.
  Enables the use of [[cuv]], [[cuge]], [[cutr]], etc. in its body.
  "
  [& body]
  `(with-engine cuda-float default-stream ~@body))

(defn cuv
  "Creates a vector using CUDA GPU engine provided to the bound [[*cuda-factory*]]
  (see [[uncomplicate.neanderthal.core/vctr]]).
  "
  ([source]
   (vctr *cuda-factory* source))
  ([x & xs]
   (cuv (cons x xs))))

(defn cuge
  "Creates a GE matrix using CUDA GPU engine provided to the bound [[*cuda-factory*]]
  (see [[uncomplicate.neanderthal.core/ge]]).
  "
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
  (see [[uncomplicate.neanderthal.core/tr]]).
  "
  ([^long n source options]
   (tr *cuda-factory* n source options))
  ([^long n arg]
   (tr *cuda-factory* n arg))
  ([arg]
   (tr *cuda-factory* arg)))

(defn cusy
  "Creates a SY matrix using CUDA GPU engine provided to the bound [[*cuda-factory*]]
  (see [[uncomplicate.neanderthal.core/sy]]).
  "
  ([^long n source options]
   (sy *cuda-factory* n source options))
  ([^long n arg]
   (sy *cuda-factory* n arg))
  ([arg]
   (sy *cuda-factory* arg)))
