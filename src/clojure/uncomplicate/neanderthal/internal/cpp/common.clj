;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://openpsource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.common
  (:require [uncomplicate.clojure-cpp :as cpp]
            [uncomplicate.neanderthal.block :refer [buffer]])
  (:import [org.bytedeco.javacpp Pointer FloatPointer DoublePointer IntPointer]
           [uncomplicate.neanderthal.internal.api Block]))

(defn float-ptr
  (^FloatPointer [^Block x]
   (cpp/float-ptr (.buffer x)))
  (^FloatPointer [^Block x ^long i]
   (.getPointer ^Pointer (.buffer x) FloatPointer (max 0 i))))

(defn double-ptr
  (^DoublePointer [^Block x]
   (cpp/double-ptr (.buffer x)))
  (^DoublePointer [^Block x ^long i]
   (.getPointer ^Pointer (.buffer x) DoublePointer (max 0 i))))

(defn int-ptr ^IntPointer [^Block x]
  (cpp/int-ptr (.buffer x)))
