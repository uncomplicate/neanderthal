;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.native
  "Specialized constructors that use native CPU engine by default. A convenience over agnostic
  [[uncomplicate.neanderthal.core]] functions."
  (:require [uncomplicate.neanderthal.core :refer [vctr ge tr sy gb tb sb]]
            [uncomplicate.neanderthal.internal.host.mkl :refer [mkl-float mkl-double mkl-int mkl-long]]))

;; ============ Creating real constructs  ==============

(def ^{:doc "Default single-precision floating point native factory"}
  native-float mkl-float)

(def ^{:doc "Default double-precision floating point native factory"}
  native-double mkl-double)

(def ^{:doc "Default integer native factory"}
  native-int mkl-int)

(def ^{:doc "Default long native factory"}
  native-long mkl-long)

(defn iv
  "Creates a vector using integer native CPU engine (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr mkl-int source))
  ([x & xs]
   (iv (cons x xs))))

(defn lv
  "Creates a vector using long CPU engine (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr mkl-long source))
  ([x & xs]
   (lv (cons x xs))))

(defn fv
  "Creates a vector using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr mkl-float source))
  ([x & xs]
   (fv (cons x xs))))

(defn dv
  "Creates a vector using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr mkl-double source))
  ([x & xs]
   (dv (cons x xs))))

(defn fge
  "Creates a GE matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/ge]])."
  ([^long m ^long n source options]
   (ge mkl-float m n source options))
  ([^long m ^long n arg]
   (ge mkl-float m n arg))
  ([^long m ^long n]
   (ge mkl-float m n))
  ([a]
   (ge mkl-float a)))

(defn dge
  "Creates a GE matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/ge]])."
  ([^long m ^long n source options]
   (ge mkl-double m n source options))
  ([^long m ^long n arg]
   (ge mkl-double m n arg))
  ([^long m ^long n]
   (ge mkl-double m n))
  ([a]
   (ge mkl-double a)))

(defn ftr
  "Creates a TR matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tr]])."
  ([^long n source options]
   (tr mkl-float n source options))
  ([^long n arg]
   (tr mkl-float n arg))
  ([arg]
   (tr mkl-float arg)))

(defn dtr
  "Creates a TR matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tr]])."
  ([^long n source options]
   (tr mkl-double n source options))
  ([^long n arg]
   (tr mkl-double n arg))
  ([arg]
   (tr mkl-double arg)))

(defn fsy
  "Creates a SY matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sy]])."
  ([^long n source options]
   (sy mkl-float n source options))
  ([^long n arg]
   (sy mkl-float n arg))
  ([arg]
   (sy mkl-float arg)))

(defn dsy
  "Creates a SY matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sy]])."
  ([^long n source options]
   (sy mkl-double n source options))
  ([^long n arg]
   (sy mkl-double n arg))
  ([arg]
   (sy mkl-double arg)))

(defn fgb
  "Creates a GB matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([m n kl ku source options]
   (gb mkl-float m n kl ku source options))
  ([m n kl ku arg]
   (gb mkl-float m n kl ku arg))
  ([m n kl ku]
   (gb mkl-float m n kl ku))
  ([arg]
   (gb mkl-float arg)))

(defn dgb
  "Creates a GB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([m n kl ku source options]
   (gb mkl-double m n kl ku source options))
  ([m n kl ku arg]
   (gb mkl-double m n kl ku arg))
  ([m n kl ku]
   (gb mkl-double m n kl ku))
  ([arg]
   (gb mkl-double arg)))

(defn ftb
  "Creates a TB matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([^long n ^long k source options]
   (tb mkl-float n k source options))
  ([^long n ^long k arg]
   (tb mkl-float n k arg))
  ([arg]
   (tb mkl-float arg)))

(defn dtb
  "Creates a TB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([^long n ^long k source options]
   (tb mkl-double n k source options))
  ([^long n ^long k arg]
   (tb mkl-double n k arg))
  ([arg]
   (tb mkl-double arg)))

(defn fsb
  "Creates a SB matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sb]])."
  ([^long n ^long k source options]
   (sb mkl-float n k source options))
  ([^long n ^long k arg]
   (sb mkl-float n k arg))
  ([arg]
   (sb mkl-float arg)))

(defn dsb
  "Creates a SB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sb]])."
  ([^long n ^long k source options]
   (sb mkl-double n k source options))
  ([^long n ^long k arg]
   (sb mkl-double n k arg))
  ([arg]
   (sb mkl-double arg)))
