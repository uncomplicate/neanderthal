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
  (:require [uncomplicate.commons.utils :refer [dragan-says-ex channel]]
            [uncomplicate.neanderthal.core :refer [vctr ge tr sy gb tb sb tp sp gd gt dt st]]
            [uncomplicate.neanderthal.internal.host
             [buffer-block :refer [map-channel]]
             [mkl :refer [mkl-float mkl-double mkl-int mkl-long mkl-short mkl-byte]]])
  (:import java.nio.channels.FileChannel))

;; ============ Creating real constructs  ==============

(def ^{:doc "Default single-precision floating point native factory"}
  native-float mkl-float)

(def ^{:doc "Default double-precision floating point native factory"}
  native-double mkl-double)

(def ^{:doc "Default integer native factory"}
  native-int mkl-int)

(def ^{:doc "Default long native factory"}
  native-long mkl-long)

(def ^{:doc "Default short native factory"}
  native-short mkl-short)

(def ^{:doc "Default byte native factory"}
  native-byte mkl-byte)

(defn factory-by-type [data-type]
  (case data-type
    :float native-float
    :int native-int
    :double native-double
    :long native-long
    :short native-short
    :byte native-byte
    :uint8 native-byte
    (cond
      (= Float/TYPE data-type) native-float
      (= Double/TYPE data-type) native-double
      (= Integer/TYPE data-type) native-int
      (= Long/TYPE data-type) native-long
      (= Short/TYPE data-type) native-short
      (= Byte/TYPE data-type) native-byte
      (= float data-type) native-float
      (= double data-type) native-double
      (= int data-type) native-int
      (= long data-type) native-long
      (= short data-type) native-short
      (= byte data-type) native-byte
      :default (dragan-says-ex "You requested a factory for an unsupported data type."
                               {:requested data-type
                                :available [:float :int :double :long :short :byte
                                            Float/TYPE Double/TYPE
                                            Integer/TYPE Long/TYPE Short/TYPE Byte/TYPE]}))))

(defn iv
  "Creates a vector using integer native CPU engine (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr native-int source))
  ([x & xs]
   (iv (cons x xs))))

(defn lv
  "Creates a vector using long CPU engine (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr native-long source))
  ([x & xs]
   (lv (cons x xs))))

(defn sv
  "Creates a vector using short native CPU engine (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr native-short source))
  ([x & xs]
   (sv (cons x xs))))

(defn bv
  "Creates a vector using byte native CPU engine (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr native-byte source))
  ([x & xs]
   (bv (cons x xs))))

(defn fv
  "Creates a vector using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr native-float source))
  ([x & xs]
   (fv (cons x xs))))

(defn dv
  "Creates a vector using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/vctr]])."
  ([source]
   (vctr native-double source))
  ([x & xs]
   (dv (cons x xs))))

(defn fge
  "Creates a GE matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/ge]])."
  ([^long m ^long n source options]
   (ge native-float m n source options))
  ([^long m ^long n arg]
   (ge native-float m n arg))
  ([^long m ^long n]
   (ge native-float m n))
  ([a]
   (ge native-float a)))

(defn dge
  "Creates a GE matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/ge]])."
  ([^long m ^long n source options]
   (ge native-double m n source options))
  ([^long m ^long n arg]
   (ge native-double m n arg))
  ([^long m ^long n]
   (ge native-double m n))
  ([a]
   (ge native-double a)))

(defn ftr
  "Creates a TR matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tr]])."
  ([^long n source options]
   (tr native-float n source options))
  ([^long n arg]
   (tr native-float n arg))
  ([arg]
   (tr native-float arg)))

(defn dtr
  "Creates a TR matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tr]])."
  ([^long n source options]
   (tr native-double n source options))
  ([^long n arg]
   (tr native-double n arg))
  ([arg]
   (tr native-double arg)))

(defn fsy
  "Creates a SY matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sy]])."
  ([^long n source options]
   (sy native-float n source options))
  ([^long n arg]
   (sy native-float n arg))
  ([arg]
   (sy native-float arg)))

(defn dsy
  "Creates a SY matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sy]])."
  ([^long n source options]
   (sy native-double n source options))
  ([^long n arg]
   (sy native-double n arg))
  ([arg]
   (sy native-double arg)))

(defn fgb
  "Creates a GB matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([m n kl ku source options]
   (gb native-float m n kl ku source options))
  ([m n kl ku arg]
   (gb native-float m n kl ku arg))
  ([m n arg]
   (gb native-float m n arg))
  ([m n kl ku]
   (gb native-float m n kl ku))
  ([m n]
   (gb native-float m n))
  ([arg]
   (gb native-float arg)))

(defn dgb
  "Creates a GB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([m n kl ku source options]
   (gb native-double m n kl ku source options))
  ([m n kl ku arg]
   (gb native-double m n kl ku arg))
  ([m n arg]
   (gb native-double m n arg))
  ([m n kl ku]
   (gb native-double m n kl ku))
  ([m n]
   (gb native-double m n))
  ([arg]
   (gb native-double arg)))

(defn ftb
  "Creates a TB matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([n k source options]
   (tb native-float n k source options))
  ([n k arg]
   (tb native-float n k arg))
  ([^long n arg]
   (tb native-float n arg))
  ([source]
   (tb native-float source)))

(defn dtb
  "Creates a TB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tb]])."
  ([n k source options]
   (tb native-double n k source options))
  ([n k arg]
   (tb native-double n k arg))
  ([^long n arg]
   (tb native-double n arg))
  ([source]
   (tb native-double source)))

(defn fsb
  "Creates a SB matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sb]])."
  ([n k source options]
   (sb native-float n k source options))
  ([n k arg]
   (sb native-float n k arg))
  ([^long n arg]
   (sb native-float n arg))
  ([source]
   (sb native-float source)))

(defn dsb
  "Creates a SB matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sb]])."
  ([n k source options]
   (sb native-double n k source options))
  ([n k arg]
   (sb native-double n k arg))
  ([^long n arg]
   (sb native-double n arg))
  ([source]
   (sb native-double source)))

(defn ftp
  "Creates a TP matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tp]])."
  ([^long n source options]
   (tp native-float n source options))
  ([^long n arg]
   (tp native-float n arg))
  ([source]
   (tp native-float source)))

(defn dtp
  "Creates a TP matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/tp]])."
  ([^long n source options]
   (tp native-double n source options))
  ([^long n arg]
   (tp native-double n arg))
  ([source]
   (tp native-double source)))

(defn fsp
  "Creates a SP matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sp]])."
  ([^long n source options]
   (sp native-float n source options))
  ([^long n arg]
   (sp native-float n arg))
  ([source]
   (sp native-float source)))

(defn dsp
  "Creates a SP matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/sp]])."
  ([^long n source options]
   (sp native-double n source options))
  ([^long n arg]
   (sp native-double n arg))
  ([source]
   (sp native-double source)))

(defn fgd
  "Creates a GD (diagonal) matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/gd]])."
  ([^long n source options]
   (gd native-float n source options))
  ([^long n arg]
   (gd native-float n arg))
  ([source]
   (gd native-float source)))

(defn dgd
  "Creates a GD (diagonal) matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/gd]])."
  ([^long n source options]
   (gd native-double n source options))
  ([^long n arg]
   (gd native-double n arg))
  ([source]
   (gd native-double source)))

(defn fgt
  "Creates a GT (tridiagonal) matrix using single precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/gt]])."
  ([^long n source options]
   (gt native-float n source options))
  ([^long n arg]
   (gt native-float n arg))
  ([source]
   (gt native-float source)))

(defn dgt
  "Creates a GT (tridiagonal) matrix using double precision floating point native CPU engine
  (see [[uncomplicate.neanderthal.core/gt]])."
  ([^long n source options]
   (gt native-double n source options))
  ([^long n arg]
   (gt native-double n arg))
  ([source]
   (gt native-double source)))

(defn fdt
  "Creates a DT (diagonally dominant tridiagonal) matrix using single precision floating point
  native CPU engine (see [[uncomplicate.neanderthal.core/dt]])."
  ([^long n source options]
   (dt native-float n source options))
  ([^long n arg]
   (dt native-float n arg))
  ([source]
   (dt native-float source)))

(defn ddt
  "Creates a DT (diagonally dominant tridiagonal) matrix using double precision floating point
  native CPU engine (see [[uncomplicate.neanderthal.core/dt]])."
  ([^long n source options]
   (dt native-double n source options))
  ([^long n arg]
   (dt native-double n arg))
  ([source]
   (dt native-double source)))

(defn fst
  "Creates a ST (symmetric positive definite tridiagonal) matrix using single precision
  floating point native CPU engine (see [[uncomplicate.neanderthal.core/st]])."
  ([^long n source options]
   (st native-float n source options))
  ([^long n arg]
   (st native-float n arg))
  ([source]
   (st native-float source)))

(defn dst
  "Creates a ST (symmetric positive definite tridiagonal) matrix using double precision
  floating point native CPU engine (see [[uncomplicate.neanderthal.core/st]])."
  ([^long n source options]
   (st native-double n source options))
  ([^long n arg]
   (st native-double n arg))
  ([source]
   (st native-double source)))

(defn map-file
  "Maps a file or file channel to a vector of size `n`.
  Supported `flag`s are: `:read-write`, `:read` (or `:read-only`),
  `:private` (or `:copy-on-write`)."
  ([fact file n flag offset-bytes]
   (let [channel (if (instance? FileChannel file) file (channel file))]
     (map-channel fact channel n flag offset-bytes)))
  ([fact file n flag]
   (map-file fact file n flag 0))
  ([fact file n-or-flag]
   (let [channel (if (instance? FileChannel file) file (channel file))]
     (map-channel fact channel n-or-flag)))
  ([fact file]
   (let [channel (if (instance? FileChannel file) file (channel file))]
     (map-channel channel file)))
  ([file]
   (map-file native-float file)))
