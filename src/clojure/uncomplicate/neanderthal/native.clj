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
  (:require [uncomplicate.neanderthal.core :refer [vctr ge tr sy gb tb sb tp sp gd gt dt st]]
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
  ([m n arg]
   (gb mkl-float m n arg))
  ([m n kl ku]
   (gb mkl-float m n kl ku))
  ([m n]
   (gb mkl-float m n))
  ([arg]
   (gb mkl-float arg)))

(defn dgb
  "Creates a GB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([m n kl ku source options]
   (gb mkl-double m n kl ku source options))
  ([m n kl ku arg]
   (gb mkl-double m n kl ku arg))
  ([m n arg]
   (gb mkl-double m n arg))
  ([m n kl ku]
   (gb mkl-double m n kl ku))
  ([m n]
   (gb mkl-double m n))
  ([arg]
   (gb mkl-double arg)))

(defn ftb
  "Creates a TB matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([n k source options]
   (tb mkl-float n k source options))
  ([n k arg]
   (tb mkl-float n k arg))
  ([^long n arg]
   (tb mkl-float n arg))
  ([source]
   (tb mkl-float source)))

(defn dtb
  "Creates a TB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([n k source options]
   (tb mkl-double n k source options))
  ([n k arg]
   (tb mkl-double n k arg))
  ([^long n arg]
   (tb mkl-double n arg))
  ([source]
   (tb mkl-double source)))

(defn fsb
  "Creates a SB matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sb]])."
  ([n k source options]
   (sb mkl-float n k source options))
  ([n k arg]
   (sb mkl-float n k arg))
  ([^long n arg]
   (sb mkl-float n arg))
  ([source]
   (sb mkl-float source)))

(defn dsb
  "Creates a SB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sb]])."
  ([n k source options]
   (sb mkl-double n k source options))
  ([n k arg]
   (sb mkl-double n k arg))
  ([^long n arg]
   (sb mkl-double n arg))
  ([source]
   (sb mkl-double source)))

(defn ftp
  "Creates a TP matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tp]])."
  ([^long n source options]
   (tp mkl-float n source options))
  ([^long n arg]
   (tp mkl-float n arg))
  ([source]
   (tp mkl-float source)))

(defn dtp
  "Creates a TP matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tp]])."
  ([^long n source options]
   (tp mkl-double n source options))
  ([^long n arg]
   (tp mkl-double n arg))
  ([source]
   (tp mkl-double source)))

(defn fsp
  "Creates a SP matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sp]])."
  ([^long n source options]
   (sp mkl-float n source options))
  ([^long n arg]
   (sp mkl-float n arg))
  ([source]
   (sp mkl-float source)))

(defn dsp
  "Creates a SP matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sp]])."
  ([^long n source options]
   (sp mkl-double n source options))
  ([^long n arg]
   (sp mkl-double n arg))
  ([source]
   (sp mkl-double source)))

(defn fgd
  "Creates a GD (diagonal) matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/gd]])."
  ([^long n source options]
   (gd mkl-float n source options))
  ([^long n arg]
   (gd mkl-float n arg))
  ([source]
   (gd mkl-float source)))

(defn dgd
  "Creates a GD (diagonal) matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/gd]])."
  ([^long n source options]
   (gd mkl-double n source options))
  ([^long n arg]
   (gd mkl-double n arg))
  ([source]
   (gd mkl-double source)))

(defn fgt
  "Creates a GT (tridiagonal) matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/gt]])."
  ([^long n source options]
   (gt mkl-float n source options))
  ([^long n arg]
   (gt mkl-float n arg))
  ([source]
   (gt mkl-float source)))

(defn dgt
  "Creates a GT (tridiagonal) matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/gt]])."
  ([^long n source options]
   (gt mkl-double n source options))
  ([^long n arg]
   (gt mkl-double n arg))
  ([source]
   (gt mkl-double source)))

(defn fdt
  "Creates a DT (diagonally dominant tridiagonal) matrix using single precision floating point
  native CPU engine (see [[uncomplicate.neanderthal.core/dt]])."
  ([^long n source options]
   (dt mkl-float n source options))
  ([^long n arg]
   (dt mkl-float n arg))
  ([source]
   (dt mkl-float source)))

(defn ddt
  "Creates a DT (diagonally dominant tridiagonal) matrix using double precision floating point
  native CPU engine (see [[uncomplicate.neanderthal.core/dt]])."
  ([^long n source options]
   (dt mkl-double n source options))
  ([^long n arg]
   (dt mkl-double n arg))
  ([source]
   (dt mkl-double source)))

(defn fst
  "Creates a ST (symmetric positive definite tridiagonal) matrix using single precision
  floating point native CPU engine (see [[uncomplicate.neanderthal.core/st]])."
  ([^long n source options]
   (st mkl-float n source options))
  ([^long n arg]
   (st mkl-float n arg))
  ([source]
   (st mkl-float source)))

(defn dst
  "Creates a ST (symmetric positive definite tridiagonal) matrix using double precision
  floating point native CPU engine (see [[uncomplicate.neanderthal.core/st]])."
  ([^long n source options]
   (st mkl-double n source options))
  ([^long n arg]
   (st mkl-double n arg))
  ([source]
   (st mkl-double source)))
