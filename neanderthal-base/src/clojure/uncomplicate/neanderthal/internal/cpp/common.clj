;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://openpsource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.cpp.common
  (:require [uncomplicate.clojure-cpp :as cpp :refer [float-pointer double-pointer]]
            [uncomplicate.neanderthal.internal
             [constants :refer :all]
             [navigation :refer [full-storage]]]
            [uncomplicate.neanderthal.internal.cpp.structures
             :refer [->RealBlockVector ->RealGEMatrix]])
  (:import [org.bytedeco.javacpp Pointer FloatPointer DoublePointer IntPointer]
           [uncomplicate.neanderthal.internal.api Block]))

(defn float-ptr
  (^FloatPointer [^Block x]
   (cpp/float-ptr (.buffer x)))
  (^FloatPointer [^Block x ^long i]
   (.getPointer ^Pointer (.buffer x) FloatPointer i)))

(defn double-ptr
  (^DoublePointer [^Block x]
   (cpp/double-ptr (.buffer x)))
  (^DoublePointer [^Block x ^long i]
   (.getPointer ^Pointer (.buffer x) DoublePointer i)))

(defn int-ptr ^IntPointer [^Block x]
  (cpp/int-ptr (.buffer x)))

(defmacro byte-float [x]
  `(let [b# (ByteBuffer/allocate Float/BYTES)
         x# (byte ~x)]
     (.put b# 0 x#)
     (.put b# 1 x#)
     (.put b# 2 x#)
     (.put b# 3 x#)
     (.getFloat b# 0)))

(defmacro short-float [x]
  `(let [b# (ByteBuffer/allocate Float/BYTES)
         x# (short ~x)]
     (.putShort b# 0 x#)
     (.putShort b# 1 x#)
     (.getFloat b# 0)))

(defmacro long-double [x]
  `(Double/longBitsToDouble ~x))

(defmacro int-float [x]
  `(Float/intBitsToFloat ~x))

(def ones-float (->RealBlockVector nil nil nil true (float-pointer [1.0]) 1 0))
(def ones-double (->RealBlockVector nil nil nil true (double-pointer [1.0]) 1 0))

(def zero-float (->RealBlockVector nil nil nil true (float-pointer [0.0]) 1 0))
(def zero-double (->RealBlockVector nil nil nil true (double-pointer [0.0]) 1 0))

(def ge-ones-float (->RealGEMatrix nil (full-storage true 0 0 Integer/MAX_VALUE)
                                nil nil nil nil true (float-pointer [1.0]) 0 0))
(def ge-ones-double (->RealGEMatrix nil (full-storage true 0 0 Integer/MAX_VALUE)
                                 nil nil nil nil true (double-pointer [1.0]) 0 0))
(def ge-zero-float (->RealGEMatrix nil (full-storage true 0 0 Integer/MAX_VALUE)
                                nil nil nil nil true (float-pointer [0.0]) 0 0))
(def ge-zero-double (->RealGEMatrix nil (full-storage true 0 0 Integer/MAX_VALUE)
                                 nil nil nil nil true (double-pointer [0.0]) 0 0))
