;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.mkl
  (:require [uncomplicate.commons
             [core :refer [with-release let-release info Releaseable release]]
             [utils :refer [dragan-says-ex generate-seed direct-buffer]]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer [dim]]
             [math :refer [f=] :as math]
             [block :refer [create-data-source initialize]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all]
             [navigation :refer [full-storage]]
             [common :refer [check-stride check-eq-navigators real-accessor]]]
            [uncomplicate.neanderthal.internal.host
             [buffer-block :refer :all]
             [cblas :refer :all]
             [lapack :refer :all]])
  (:import [uncomplicate.neanderthal.internal.host CBLAS MKL LAPACK]
           [java.nio ByteBuffer FloatBuffer DoubleBuffer LongBuffer IntBuffer DirectByteBuffer
            DirectFloatBufferU DirectDoubleBufferU DirectLongBufferU DirectIntBufferU]
           [uncomplicate.neanderthal.internal.api DataAccessor RealBufferAccessor
            Block RealVector Region LayoutNavigator DenseStorage RealNativeMatrix]
           [uncomplicate.neanderthal.internal.host.buffer_block IntegerBlockVector RealBlockVector
            RealGEMatrix RealUploMatrix RealBandedMatrix RealPackedMatrix RealDiagonalMatrix]))

(defn ^:private not-available [kind]
  (throw (UnsupportedOperationException. "Operation not available for %s matrix")))

(def ^{:no-doc true :const true} INTEGER_UNSUPPORTED_MSG
  "\nInteger BLAS operations are not supported. Please transform data to float or double.\n")

(def ^{:no-doc true :const true} SHORT_UNSUPPORTED_MSG
  "BLAS operation on short vectors are supported only on dimensions divisible by 2 (short) or 4 (byte).")

;; =========== MKL RNG routines =========================================================

(defmacro with-mkl-check [expr res]
  ` (let [err# ~expr]
      (if (zero? err#)
        ~res
        (throw (ex-info "MKL error." {:error-code err#})))))

(defn ^:private params-buffer [^RealBufferAccessor da a b]
  (let-release [buf (create-data-source da 2)]
    (.set da buf 0 a)
    (.set da buf 1 b)
    buf))

(defmacro vector-random
  ([method stream a b x]
   `(if (< 0 (.dim ~x))
      (if (= 1 (.stride ~x))
        (with-mkl-check (~method ~stream (.dim ~x) (.buffer ~x) (.offset ~x) ~a ~b)
          ~x)
        (dragan-says-ex "This engine cannot generate random entries in host vectors with stride. Sorry."
                        {:v (info ~x)}))
      ~x)))

(defmacro matrix-random
  ([method stream a b x]
   `(if (< 0 (.dim ~x))
      (if (.isGapless (storage ~x))
        (with-mkl-check (~method ~stream (.dim ~x) (.buffer ~x) (.offset ~x) ~a ~b)
          ~x)
        (dragan-says-ex "This engine cannot generate random entries in host matrices with stride. Sorry."
                        {:v (info ~x)}))
      ~x)))

(defn create-stream-ars5 [seed]
  (let-release [stream (direct-buffer Long/BYTES)]
    (with-mkl-check (MKL/vslNewStreamARS5 (unchecked-int seed) stream)
      stream)))

(def ^:private default-rng-stream (create-stream-ars5 (generate-seed)))

;; =========== MKL-specific routines ====================================================

(defmacro ge-copy [method a b]
  `(if (< 0 (.dim ~a))
     (let [stor-b# (full-storage ~b)
           no-trans# (= (navigator ~a) (navigator ~b))
           rows# (if no-trans# (.sd stor-b#) (.fd stor-b#))
           cols# (if no-trans# (.fd stor-b#) (.sd stor-b#))]
       (~method (int \C) (int (if no-trans# \N \T)) rows# cols#
        1.0 (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.ld stor-b#))
       ~b)
     ~b))

(defmacro ge-scal [method alpha a]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)]
       (~method (int \c) (int \n) (.sd stor#) (.fd stor#)
        ~alpha (.buffer ~a) (.offset ~a) (.ld stor#) (.ld stor#))
       ~a)
     ~a))

(defmacro ge-trans [method a]
  `(if (< 0 (.dim ~a))
     (let [stor# (full-storage ~a)]
       (if (.isGapless stor#)
         (~method (int \c) (int \t) (.sd stor#) (.fd stor#)
          1.0 (.buffer ~a) (.offset ~a) (.ld stor#) (.fd stor#))
         (dragan-says-ex "You can not hard-transpose the content of a matrix with a gap in memory. Sorry."
                         {:a (info ~a)}))
       ~a)
     ~a))

(defmacro ge-axpby [method alpha a beta b]
  `(if (< 0 (.dim ~a))
     (let [nav-b# (navigator ~b)]
       (~method (int (if (.isColumnMajor nav-b#) \C \R))
        (int (if (= (navigator ~a) nav-b#) \n \t)) (int \n) (.mrows ~b) (.ncols ~b)
        ~alpha (.buffer ~a) (.offset ~a) (.stride ~a) ~beta (.buffer ~b) (.offset ~b) (.stride ~b)
        (.buffer ~b) (.offset ~b) (.stride ~b))
       ~b)
     ~b))

(defmacro gd-sv [vdiv-method sv-method a b]
  `(let [n-a# (.ncols ~a)
         n-b# (.ncols ~b)
         nav-b# (navigator ~b)
         stor-b# (storage ~b)
         buff-a# (.buffer ~a)
         ofst-a# (.offset ~a)
         buff-b# (.buffer ~b)
         ofst-b# (.offset ~b)
         strd-b# (.stride ~b)]
     (if (.isColumnMajor nav-b#)
       (dotimes [j# n-b#]
         (~vdiv-method n-a# buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) buff-a# ofst-a#
          buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#))))
       (dotimes [j# (.ncols ~b)]
         (~sv-method CBLAS/ORDER_ROW_MAJOR CBLAS/UPLO_LOWER CBLAS/TRANSPOSE_NO_TRANS CBLAS/DIAG_NON_UNIT n-a# 0
          buff-a# ofst-a# 1 buff-b# (+ ofst-b# (.index nav-b# stor-b# 0 j#)) strd-b#)))
     ~b))

(defmacro gd-tri [method a]
  `(do
     (~method (.ncols ~a) (.buffer ~a) (.offset ~a) (.buffer ~a) (.offset ~a))
     ~a))

;; ------------ MKL specific vector functions -------------------------------------

(defmacro vector-math
  ([method a y]
   ` (do
       (check-stride ~a ~y)
       (~method (.dim ~a) (.buffer ~a) (.offset ~a) (.buffer ~y) (.offset ~y))
       ~y))
  ([method a b y]
   `(do
      (check-stride ~a ~b ~y)
      (~method (.dim ~a) (.buffer ~a) (.offset ~a) (.buffer ~b) (.offset ~b) (.buffer ~y) (.offset ~y))
      ~y)))

(defmacro vector-powx [method a b y]
  `(do
     (check-stride ~a ~y)
     (~method (.dim ~a) (.buffer ~a) (.offset ~a) ~b (.buffer ~y) (.offset ~y))
     ~y))

(defmacro vector-linear-frac [method a b scalea shifta scaleb shiftb y]
  `(do
     (check-stride ~a ~b ~y)
     (~method (.dim ~a) (.buffer ~a) (.offset ~a) (.buffer ~b) (.offset ~b)
      ~scalea ~shifta ~scaleb ~shiftb (.buffer ~y) (.offset ~y))
     ~y))

(defmacro ^:private full-matching-map
  ([a b len offset-a offset-b expr-direct expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          fd-a# (.fd stor-a#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)]
      (check-eq-navigators ~a ~b)
      (if (and (.isGapless stor-a#) (.isGapless stor-b#))
        ~expr-direct
        (dotimes [j# fd-a#]
          (let [start# (.start nav-a# reg# j#)
                ~len (- (.end nav-a# reg# j#) start#)
                ~offset-a (+ offset-a# (.index stor-a# start# j#))
                ~offset-b (+ offset-b# (.index stor-b# start# j#))]
            ~expr)))))
  ([a b c len offset-a offset-b offset-c expr-direct expr]
   `(let [nav-a# (navigator ~a)
          nav-b# (navigator ~b)
          nav-c# (navigator ~c)
          reg# (region ~b)
          stor-a# (full-storage ~a)
          stor-b# (full-storage ~b)
          stor-c# (full-storage ~c)
          fd-a# (.fd stor-a#)
          offset-a# (.offset ~a)
          offset-b# (.offset ~b)
          offset-c# (.offset ~c)]
      (check-eq-navigators ~a ~b ~c)
      (if (and (.isGapless stor-a#) (.isGapless stor-b#) (.isGapless stor-c#))
        ~expr-direct
        (dotimes [j# fd-a#]
          (let [start# (.start nav-a# reg# j#)
                ~len (- (.end nav-a# reg# j#) start#)
                ~offset-a (+ offset-a# (.index stor-a# start# j#))
                ~offset-b (+ offset-b# (.index stor-b# start# j#))
                ~offset-c (+ offset-c# (.index stor-c# start# j#))]
            ~expr))))))

(defmacro matrix-math
  ([method a y]
   `(do
      (when (< 0 (.dim ~a))
        (let [buff-a# (.buffer ~a)
              buff-y# (.buffer ~y)]
          (full-matching-map ~a ~y len# offset-a# offset-y#
                             (~method (.dim ~a) buff-a# (.offset ~a) buff-y# (.offset ~y))
                             (~method len# buff-a# offset-a# buff-y# offset-y#))))
      ~y))
  ([method a b y]
   `(do
      (when (< 0 (.dim ~a))
        (let [buff-a# (.buffer ~a)
              buff-b# (.buffer ~b)
              buff-y# (.buffer ~y)]
          (full-matching-map ~a ~b ~y len# offset-a# offset-b# offset-y#
                             (~method (.dim ~a) buff-a# (.offset ~a) buff-b# (.offset ~b) buff-y# (.offset ~y))
                             (~method len# buff-a# offset-a# buff-b# offset-b# buff-y# offset-y#))))
      ~y)))

(defmacro matrix-powx [method a b y]
  `(do
     (when (< 0 (.dim ~a))
       (let [buff-a# (.buffer ~a)
             buff-y# (.buffer ~y)]
         (full-matching-map ~a ~y len# offset-a# offset-y#
                            (~method (.dim ~a) buff-a# (.offset ~a) ~b  buff-y# (.offset ~y))
                            (~method len# buff-a# offset-a# ~b buff-y# offset-y#))))
     ~y))

(defmacro matrix-linear-frac [method a b scalea shifta scaleb shiftb y]
  `(do
     (when (< 0 (.dim ~a))
       (let [buff-a# (.buffer ~a)
             buff-b# (.buffer ~b)
             buff-y# (.buffer ~y)]
         (full-matching-map ~a ~b ~y len# offset-a# offset-b# offset-y#
                            (~method (.dim ~a) buff-a# (.offset ~a) buff-b# (.offset ~b)
                             ~scalea ~shifta ~scaleb ~shiftb buff-y# (.offset ~y))
                            (~method len# buff-a# offset-a# buff-b# offset-b#
                             ~scalea ~shifta ~scaleb ~shiftb buff-y# offset-y#))))
     ~y))

(defmacro packed-math
  ([method a y]
   `(do
      (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) (.buffer ~y) (.offset ~y))
      ~y))
  ([method a b y]
   `(do
      (check-eq-navigators ~a ~b)
      (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) (.buffer ~b) (.offset ~b)
       (.buffer ~y) (.offset ~y))
      ~y)))

(defmacro packed-powx [method a b y]
  `(do
     (check-eq-navigators ~a ~y)
     (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) ~b (.buffer ~y) (.offset ~y))
     ~y))

(defmacro packed-linear-frac [method a b scalea shifta scaleb shiftb y]
  `(do
     (check-eq-navigators ~a ~b ~y)
     (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) (.buffer ~b) (.offset ~b)
      ~scalea ~shifta ~scaleb ~shiftb (.buffer ~y) (.offset ~y))
     ~y))

(defmacro diagonal-math
  ([method a y]
   `(do
      (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) (.buffer ~y) (.offset ~y))
      ~y))
  ([method a b y]
   `(do
      (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) (.buffer ~b) (.offset ~b)
       (.buffer ~y) (.offset ~y))
      ~y)))

(defmacro diagonal-powx [method a b y]
  `(do
     (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) ~b (.buffer ~y) (.offset ~y))
     ~y))

(defmacro diagonal-linear-frac [method a b scalea shifta scaleb shiftb y]
  `(do
     (~method (.surface (region ~a)) (.buffer ~a) (.offset ~a) (.buffer ~b) (.offset ~b)
      ~scalea ~shifta ~scaleb ~shiftb (.buffer ~y) (.offset ~y))
     ~y))

;; ============ Delegate math functions  ============================================

(defn sigmoid-over-tanh [eng a y]
  (when-not (identical? a y) (copy eng a y))
  (linear-frac eng (tanh eng (scal eng 0.5 y) y) a 0.5 0.5 0.0 1.0 y))

(defn vector-ramp [eng a y]
  (cond (identical? a y) (fmap! math/ramp y)
        (= 1 (.stride ^Block a) (.stride ^Block y)) (fmax eng a (set-all eng 0.0 y) y)
        :else (fmap! math/ramp (copy eng a y))))

(defn matrix-ramp [eng a y]
  (if (identical? a y)
    (fmap! math/ramp y)
    (fmax eng a (set-all eng 0.0 y) y)))

(defn vector-relu [eng alpha a y]
  (cond (identical? a y) (fmap! (math/relu alpha) y)
        (= 1 (.stride ^Block a) (.stride ^Block y)) (fmax eng a (axpby eng alpha a 0.0 y) y)
        :else (fmap! (math/relu alpha) (copy eng a y))))

(defn matrix-relu [eng alpha a y]
  (if (identical? a y)
    (fmap! (math/relu alpha) y)
    (fmax eng a (axpby eng alpha a 0.0 y) y)))

(defn vector-elu [eng alpha a y]
  (cond (identical? a y) (fmap! (math/elu alpha) y)
        (= 1 (.stride ^Block a) (.stride ^Block y))
        (fmax eng a (scal eng alpha (expm1 eng (copy eng a y) y)) y)
        :else (fmap! (math/elu alpha) (copy eng a y))))

(defn matrix-elu [eng alpha a y]
  (if (identical? a y)
    (fmap! (math/elu alpha) y)
    (fmax eng a (scal eng alpha (expm1 eng (copy eng a y) y)) y)))

;; ============ Integer Vector Engines ============================================

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

(deftype LongVectorEngine []
  Blas
  (swap [_ x y]
    (vector-method CBLAS/dswap ^IntegerBlockVector x ^IntegerBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/dcopy ^IntegerBlockVector x ^IntegerBlockVector y)
    y)
  (dot [_ x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm1 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm2 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrmi [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (asum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (iamax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rot [_ x y c s]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotg [_ abcs]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotm [_ x y param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotmg [_ d1d2xy param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (scal [_ alpha x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (axpy [_ alpha x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  BlasPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer ^IntegerBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                 (.buffer ^IntegerBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
    y)
  (sum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imin [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (set-all [_ alpha x]
    (vctr-laset LAPACK/dlaset (Double/longBitsToDouble alpha) ^IntegerBlockVector x)
    x)
  (axpby [_ alpha x beta y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))

(deftype IntVectorEngine []
  Blas
  (swap [_ x y]
    (vector-method CBLAS/sswap ^IntegerBlockVector x ^IntegerBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/scopy ^IntegerBlockVector x ^IntegerBlockVector y)
    y)
  (dot [_ x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm1 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm2 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrmi [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (asum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (iamax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rot [_ x y c s]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotg [_ abcs]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotm [_ x y param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotmg [_ d1d2xy param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (scal [_ alpha x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (axpy [_ alpha x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  BlasPlus
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer ^IntegerBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                 (.buffer ^IntegerBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
    y)
  (sum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imin [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (set-all [_ alpha x]
    (vctr-laset LAPACK/slaset (Float/intBitsToFloat alpha) ^IntegerBlockVector x)
    x)
  (axpby [_ alpha x beta y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))

(deftype ShortVectorEngine []
  Blas
  (swap [_ x y]
    (check-stride x y)
    (if (= 0 (rem (dim x) 2))
      (vector-method CBLAS/sswap ^IntegerBlockVector x ^IntegerBlockVector y)
      (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim x)}))
    x)
  (copy [_ x y]
    (check-stride x y)
    (if (= 0 (rem (dim x) 2))
      (vector-method CBLAS/scopy ^IntegerBlockVector x ^IntegerBlockVector y)
      (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim x)}))
    y)
  (dot [_ x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm1 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm2 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrmi [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (asum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (iamax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rot [_ x y c s]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotg [_ abcs]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotm [_ x y param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotmg [_ d1d2xy param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (scal [_ alpha x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (axpy [_ alpha x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  BlasPlus
  (subcopy [_ x y kx lx ky]
    (check-stride x y)
    (if (= 0 (rem (long lx) 2))
      (CBLAS/scopy lx (.buffer ^IntegerBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                   (.buffer ^IntegerBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
      (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim x)}))

    y)
  (sum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imin [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (set-all [_ alpha x]
    (check-stride x)
    (if (= 0 (rem (dim x) 2))
      (vctr-laset LAPACK/slaset (short-float alpha) ^IntegerBlockVector x)
      (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim x)}))
    x)
  (axpby [_ alpha x beta y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))

(deftype ByteVectorEngine []
  Blas
  (swap [_ x y]
    (check-stride x y)
    (if (= 0 (rem (dim x) 4))
      (vector-method CBLAS/sswap ^IntegerBlockVector x ^IntegerBlockVector y)
      (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim x)}))
    x)
  (copy [_ x y]
    (check-stride x y)
    (if (= 0 (rem (dim x) 4))
      (vector-method CBLAS/scopy ^IntegerBlockVector x ^IntegerBlockVector y)
      (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim x)}))
    y)
  (dot [_ x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm1 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrm2 [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (nrmi [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (asum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (iamax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rot [_ x y c s]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotg [_ abcs]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotm [_ x y param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (rotmg [_ d1d2xy param]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (scal [_ alpha x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (axpy [_ alpha x y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  BlasPlus
  (subcopy [_ x y kx lx ky]
    (check-stride x y)
    (if (= 0 (rem (long lx) 4))
      (CBLAS/scopy lx (.buffer ^IntegerBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                   (.buffer ^IntegerBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
      (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim x)}))

    y)
  (sum [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imax [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (imin [_ x]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG)))
  (set-all [_ alpha x]
    (check-stride x)
    (if (= 0 (rem (dim x) 4))
      (vctr-laset LAPACK/slaset (byte-float alpha) ^IntegerBlockVector x)
      (dragan-says-ex SHORT_UNSUPPORTED_MSG {:dim-x (dim x)}))
    x)
  (axpby [_ alpha x beta y]
    (throw (UnsupportedOperationException. INTEGER_UNSUPPORTED_MSG))))

;; ============ Real Vector Engines ============================================

(def ^:private ones-double
  (->RealBlockVector nil nil nil true
                     (initialize double-accessor (create-data-source double-accessor 1024) 1.0)
                     1024 0 1))

(def ^:private ones-float
  (->RealBlockVector nil nil nil true
                     (initialize float-accessor (create-data-source float-accessor 1024) 1.0)
                     1024 0 1))

(deftype DoubleVectorEngine []
  Blas
  (swap [_ x y]
    (vector-method CBLAS/dswap ^RealBlockVector x ^RealBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/dcopy ^RealBlockVector x ^RealBlockVector y)
    y)
  (dot [_ x y]
    (vector-method CBLAS/ddot ^RealBlockVector x ^RealBlockVector y))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (vector-method CBLAS/dnrm2 ^RealBlockVector x))
  (nrmi [this x]
    (amax this x))
  (asum [_ x]
    (vector-method CBLAS/dasum ^RealBlockVector x))
  (iamax [_ x]
    (vector-method CBLAS/idamax ^RealBlockVector x))
  (iamin [_ x]
    (vector-method CBLAS/idamin ^RealBlockVector x))
  (rot [_ x y c s]
    (vector-rot CBLAS/drot ^RealBlockVector x ^RealBlockVector y c s)
    x)
  (rotg [_ abcs]
    (CBLAS/drotg (.buffer ^RealBlockVector abcs) (.offset ^Block abcs) (.stride ^Block abcs))
    abcs)
  (rotm [_ x y param]
    (vector-rotm CBLAS/drotm ^RealBlockVector x ^RealBlockVector y ^RealBlockVector param))
  (rotmg [_ d1d2xy param]
    (vector-rotmg CBLAS/drotmg ^RealBlockVector d1d2xy ^RealBlockVector param))
  (scal [_ alpha x]
    (CBLAS/dscal (.dim ^RealBlockVector x)
                 alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x))
    x)
  (axpy [_ alpha x y]
    (CBLAS/daxpy (.dim ^RealBlockVector x)
                 alpha  (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                 (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  BlasPlus
  (amax [_ x]
    (vector-amax ^RealBlockVector x))
  (subcopy [_ x y kx lx ky]
    (CBLAS/dcopy lx (.buffer ^RealBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                 (.buffer ^RealBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
    y)
  (sum [_ x]
    (vector-sum CBLAS/ddot ^RealBlockVector x ^RealBlockVector ones-double))
  (imax [_ x]
    (vector-imax ^RealBlockVector x))
  (imin [_ x]
    (vector-imin ^RealBlockVector x))
  (set-all [_ alpha x]
    (vctr-laset LAPACK/dlaset alpha ^RealBlockVector x))
  (axpby [_ alpha x beta y]
    (MKL/daxpby (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                beta (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  Lapack
  (srt [_ x increasing]
    (vctr-lasrt LAPACK/dlasrt ^RealBlockVector x increasing))
  VectorMath
  (sqr [_ a y]
    (vector-math MKL/vdSqr ^RealBlockVector a ^RealBlockVector y))
  (mul [_ a b y]
    (vector-math MKL/vdMul ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (div [_ a b y]
    (vector-math MKL/vdDiv ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (inv [_ a y]
    (vector-math MKL/vdInv ^RealBlockVector a ^RealBlockVector y))
  (abs [_ a y]
    (vector-math MKL/vdAbs ^RealBlockVector a ^RealBlockVector y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (vector-linear-frac MKL/vdLinearFrac ^RealBlockVector a ^RealBlockVector b
                        scalea shifta scaleb shiftb ^RealBlockVector y))
  (fmod [_ a b y]
    (vector-math MKL/vdFmod ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (frem [_ a b y]
    (vector-math MKL/vdRemainder ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (sqrt [_ a y]
    (vector-math MKL/vdSqrt ^RealBlockVector a ^RealBlockVector y))
  (inv-sqrt [_ a y]
    (vector-math MKL/vdInvSqrt ^RealBlockVector a ^RealBlockVector y))
  (cbrt [_ a y]
    (vector-math MKL/vdCbrt ^RealBlockVector a ^RealBlockVector y))
  (inv-cbrt [_ a y]
    (vector-math MKL/vdInvCbrt ^RealBlockVector a ^RealBlockVector y))
  (pow2o3 [_ a y]
    (vector-math MKL/vdPow2o3 ^RealBlockVector a ^RealBlockVector y))
  (pow3o2 [_ a y]
    (vector-math MKL/vdPow3o2 ^RealBlockVector a ^RealBlockVector y))
  (pow [_ a b y]
    (vector-math MKL/vdPow ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (powx [_ a b y]
    (vector-powx MKL/vdPowx ^RealBlockVector a b ^RealBlockVector y))
  (hypot [_ a b y]
    (vector-math MKL/vdHypot ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (exp [_ a y]
    (vector-math MKL/vdExp ^RealBlockVector a ^RealBlockVector y))
  (expm1 [_ a y]
    (vector-math MKL/vdExpm1 ^RealBlockVector a ^RealBlockVector y))
  (log [_ a y]
    (vector-math MKL/vdLn ^RealBlockVector a ^RealBlockVector y))
  (log10 [_ a y]
    (vector-math MKL/vdLog10 ^RealBlockVector a ^RealBlockVector y))
  (sin [_ a y]
    (vector-math MKL/vdSin ^RealBlockVector a ^RealBlockVector y))
  (cos [_ a y]
    (vector-math MKL/vdCos ^RealBlockVector a ^RealBlockVector y))
  (tan [_ a y]
    (vector-math MKL/vdTan ^RealBlockVector a ^RealBlockVector y))
  (sincos [_ a y z]
    (vector-math MKL/vdSinCos ^RealBlockVector a ^RealBlockVector y ^RealBlockVector z))
  (asin [_ a y]
    (vector-math MKL/vdAsin ^RealBlockVector a ^RealBlockVector y))
  (acos [_ a y]
    (vector-math MKL/vdAcos ^RealBlockVector a ^RealBlockVector y))
  (atan [_ a y]
    (vector-math MKL/vdAtan ^RealBlockVector a ^RealBlockVector y))
  (atan2 [_ a b y]
    (vector-math MKL/vdAtan2 ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (sinh [_ a y]
    (vector-math MKL/vdSinh ^RealBlockVector a ^RealBlockVector y))
  (cosh [_ a y]
    (vector-math MKL/vdCosh ^RealBlockVector a ^RealBlockVector y))
  (tanh [_ a y]
    (vector-math MKL/vdTanh ^RealBlockVector a ^RealBlockVector y))
  (asinh [_ a y]
    (vector-math MKL/vdAsinh ^RealBlockVector a ^RealBlockVector y))
  (acosh [_ a y]
    (vector-math MKL/vdAcosh ^RealBlockVector a ^RealBlockVector y))
  (atanh [_ a y]
    (vector-math MKL/vdAtanh ^RealBlockVector a ^RealBlockVector y))
  (erf [_ a y]
    (vector-math MKL/vdErf ^RealBlockVector a ^RealBlockVector y))
  (erfc [_ a y]
    (vector-math MKL/vdErfc ^RealBlockVector a ^RealBlockVector y))
  (erf-inv [_ a y]
    (vector-math MKL/vdErfInv ^RealBlockVector a ^RealBlockVector y))
  (erfc-inv [_ a y]
    (vector-math MKL/vdErfcInv ^RealBlockVector a ^RealBlockVector y))
  (cdf-norm [_ a y]
    (vector-math MKL/vdCdfNorm ^RealBlockVector a ^RealBlockVector y))
  (cdf-norm-inv [_ a y]
    (vector-math MKL/vdCdfNormInv ^RealBlockVector a ^RealBlockVector y))
  (gamma [_ a y]
    (vector-math MKL/vdGamma ^RealBlockVector a ^RealBlockVector y))
  (lgamma [_ a y]
    (vector-math MKL/vdLGamma ^RealBlockVector a ^RealBlockVector y))
  (expint1 [_ a y]
    (vector-math MKL/vdExpInt1 ^RealBlockVector a ^RealBlockVector y))
  (floor [_ a y]
    (vector-math MKL/vdFloor ^RealBlockVector a ^RealBlockVector y))
  (fceil [_ a y]
    (vector-math MKL/vdCeil ^RealBlockVector a ^RealBlockVector y))
  (trunc [_ a y]
    (vector-math MKL/vdTrunc ^RealBlockVector a ^RealBlockVector y))
  (round [_ a y]
    (vector-math MKL/vdRound ^RealBlockVector a ^RealBlockVector y))
  (modf [_ a y z]
    (vector-math MKL/vdModf ^RealBlockVector a ^RealBlockVector y ^RealBlockVector z))
  (frac [_ a y]
    (vector-math MKL/vdFrac ^RealBlockVector a ^RealBlockVector y))
  (fmin [_ a b y]
    (vector-math MKL/vdFmin ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (fmax [_ a b y]
    (vector-math MKL/vdFmax ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (copy-sign [_ a b y]
    (vector-math MKL/vdCopySign ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (vector-ramp this a y))
  (relu [this alpha a y]
    (vector-relu this alpha a y))
  (elu [this alpha a y]
    (vector-elu this alpha a y))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (vector-random MKL/vdRngUniform (or rng-stream default-rng-stream)
                   lower upper ^RealBlockVector x))
  (rand-normal [_ rng-stream mu sigma x]
    (vector-random MKL/vdRngGaussian (or rng-stream default-rng-stream)
                   mu sigma ^RealBlockVector x)))

(deftype FloatVectorEngine []
  Blas
  (swap [_ x y]
    (vector-method CBLAS/sswap ^RealBlockVector x ^RealBlockVector y)
    x)
  (copy [_ x y]
    (vector-method CBLAS/scopy ^RealBlockVector x ^RealBlockVector y)
    y)
  (dot [_ x y]
    (vector-method CBLAS/sdot ^RealBlockVector x ^RealBlockVector y))
  (nrm1 [this x]
    (asum this x))
  (nrm2 [_ x]
    (vector-method CBLAS/snrm2 ^RealBlockVector x))
  (nrmi [this x]
    (amax this x))
  (asum [_ x]
    (vector-method CBLAS/sasum ^RealBlockVector x))
  (iamax [_ x]
    (vector-method CBLAS/isamax ^RealBlockVector x))
  (iamin [_ x]
    (vector-method CBLAS/isamin ^RealBlockVector x))
  (rot [_ x y c s]
    (vector-rot CBLAS/srot ^RealBlockVector x ^RealBlockVector y c s))
  (rotg [_ abcs]
    (CBLAS/srotg (.buffer ^RealBlockVector abcs) (.offset ^Block abcs) (.stride ^Block abcs))
    abcs)
  (rotm [_ x y param]
    (vector-rotm CBLAS/srotm ^RealBlockVector x ^RealBlockVector y ^RealBlockVector param))
  (rotmg [_ d1d2xy param]
    (vector-rotmg CBLAS/srotmg ^RealBlockVector d1d2xy ^RealBlockVector param))
  (scal [_ alpha x]
    (CBLAS/sscal (.dim ^RealBlockVector x)
                 alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x))
    x)
  (axpy [_ alpha x y]
    (CBLAS/saxpy (.dim ^RealBlockVector x)
                 alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                 (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  BlasPlus
  (amax [_ x]
    (vector-amax ^RealBlockVector x))
  (subcopy [_ x y kx lx ky]
    (CBLAS/scopy lx (.buffer ^RealBlockVector x) (+ (long kx) (.offset ^Block x)) (.stride ^Block x)
                 (.buffer ^RealBlockVector y) (+ (long ky) (.offset ^Block y)) (.stride ^Block y))
    y)
  (sum [_ x]
    (vector-sum CBLAS/sdot ^RealBlockVector x ^RealBlockVector ones-float))
  (imax [_ x]
    (vector-imax ^RealBlockVector x))
  (imin [_ x]
    (vector-imin ^RealBlockVector x))
  (set-all [_ alpha x]
    (vctr-laset LAPACK/slaset alpha ^RealBlockVector x))
  (axpby [_ alpha x beta y]
    (MKL/saxpby (.dim ^RealBlockVector x) alpha (.buffer ^Block x) (.offset ^Block x) (.stride ^Block x)
                beta (.buffer ^RealBlockVector y) (.offset ^Block y) (.stride ^Block y))
    y)
  Lapack
  (srt [_ x increasing]
    (vctr-lasrt LAPACK/slasrt ^RealBlockVector x increasing))
  VectorMath
  (sqr [_ a y]
    (vector-math MKL/vsSqr ^RealBlockVector a ^RealBlockVector y))
  (mul [_ a b y]
    (vector-math MKL/vsMul ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (div [_ a b y]
    (vector-math MKL/vsDiv ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (inv [_ a y]
    (vector-math MKL/vsInv ^RealBlockVector a ^RealBlockVector y))
  (abs [_ a y]
    (vector-math MKL/vsAbs ^RealBlockVector a ^RealBlockVector y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (vector-linear-frac MKL/vsLinearFrac ^RealBlockVector a ^RealBlockVector b
                        scalea shifta scaleb shiftb ^RealBlockVector y))
  (fmod [_ a b y]
    (vector-math MKL/vsFmod ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (frem [_ a b y]
    (vector-math MKL/vsRemainder ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (sqrt [_ a y]
    (vector-math MKL/vsSqrt ^RealBlockVector a ^RealBlockVector y))
  (inv-sqrt [_ a y]
    (vector-math MKL/vsInvSqrt ^RealBlockVector a ^RealBlockVector y))
  (cbrt [_ a y]
    (vector-math MKL/vsCbrt ^RealBlockVector a ^RealBlockVector y))
  (inv-cbrt [_ a y]
    (vector-math MKL/vsInvCbrt ^RealBlockVector a ^RealBlockVector y))
  (pow2o3 [_ a y]
    (vector-math MKL/vsPow2o3 ^RealBlockVector a ^RealBlockVector y))
  (pow3o2 [_ a y]
    (vector-math MKL/vsPow3o2 ^RealBlockVector a ^RealBlockVector y))
  (pow [_ a b y]
    (vector-math MKL/vsPow ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (powx [_ a b y]
    (vector-powx MKL/vsPowx ^RealBlockVector a b ^RealBlockVector y))
  (hypot [_ a b y]
    (vector-math MKL/vsHypot ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (exp [_ a y]
    (vector-math MKL/vsExp ^RealBlockVector a ^RealBlockVector y))
  (expm1 [_ a y]
    (vector-math MKL/vsExpm1 ^RealBlockVector a ^RealBlockVector y))
  (log [_ a y]
    (vector-math MKL/vsLn ^RealBlockVector a ^RealBlockVector y))
  (log10 [_ a y]
    (vector-math MKL/vsLog10 ^RealBlockVector a ^RealBlockVector y))
  (sin [_ a y]
    (vector-math MKL/vsSin ^RealBlockVector a ^RealBlockVector y))
  (cos [_ a y]
    (vector-math MKL/vsCos ^RealBlockVector a ^RealBlockVector y))
  (tan [_ a y]
    (vector-math MKL/vsTan ^RealBlockVector a ^RealBlockVector y))
  (sincos [_ a y z]
    (vector-math MKL/vsSinCos ^RealBlockVector a ^RealBlockVector y ^RealBlockVector z))
  (asin [_ a y]
    (vector-math MKL/vsAsin ^RealBlockVector a ^RealBlockVector y))
  (acos [_ a y]
    (vector-math MKL/vsAcos ^RealBlockVector a ^RealBlockVector y))
  (atan [_ a y]
    (vector-math MKL/vsAtan ^RealBlockVector a ^RealBlockVector y))
  (atan2 [_ a b y]
    (vector-math MKL/vsAtan2 ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (sinh [_ a y]
    (vector-math MKL/vsSinh ^RealBlockVector a ^RealBlockVector y))
  (cosh [_ a y]
    (vector-math MKL/vsCosh ^RealBlockVector a ^RealBlockVector y))
  (tanh [_ a y]
    (vector-math MKL/vsTanh ^RealBlockVector a ^RealBlockVector y))
  (asinh [_ a y]
    (vector-math MKL/vsAsinh ^RealBlockVector a ^RealBlockVector y))
  (acosh [_ a y]
    (vector-math MKL/vsAcosh ^RealBlockVector a ^RealBlockVector y))
  (atanh [_ a y]
    (vector-math MKL/vsAtanh ^RealBlockVector a ^RealBlockVector y))
  (erf [_ a y]
    (vector-math MKL/vsErf ^RealBlockVector a ^RealBlockVector y))
  (erfc [_ a y]
    (vector-math MKL/vsErfc ^RealBlockVector a ^RealBlockVector y))
  (erf-inv [_ a y]
    (vector-math MKL/vsErfInv ^RealBlockVector a ^RealBlockVector y))
  (erfc-inv [_ a y]
    (vector-math MKL/vsErfcInv ^RealBlockVector a ^RealBlockVector y))
  (cdf-norm [_ a y]
    (vector-math MKL/vsCdfNorm ^RealBlockVector a ^RealBlockVector y))
  (cdf-norm-inv [_ a y]
    (vector-math MKL/vsCdfNormInv ^RealBlockVector a ^RealBlockVector y))
  (gamma [_ a y]
    (vector-math MKL/vsGamma ^RealBlockVector a ^RealBlockVector y))
  (lgamma [_ a y]
    (vector-math MKL/vsLGamma ^RealBlockVector a ^RealBlockVector y))
  (expint1 [_ a y]
    (vector-math MKL/vsExpInt1 ^RealBlockVector a ^RealBlockVector y))
  (floor [_ a y]
    (vector-math MKL/vsFloor ^RealBlockVector a ^RealBlockVector y))
  (fceil [_ a y]
    (vector-math MKL/vsCeil ^RealBlockVector a ^RealBlockVector y))
  (trunc [_ a y]
    (vector-math MKL/vsTrunc ^RealBlockVector a ^RealBlockVector y))
  (round [_ a y]
    (vector-math MKL/vsRound ^RealBlockVector a ^RealBlockVector y))
  (modf [_ a y z]
    (vector-math MKL/vsModf ^RealBlockVector a ^RealBlockVector y ^RealBlockVector z))
  (frac [_ a y]
    (vector-math MKL/vsFrac ^RealBlockVector a ^RealBlockVector y))
  (fmin [_ a b y]
    (vector-math MKL/vsFmin ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (fmax [_ a b y]
    (vector-math MKL/vsFmax ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (copy-sign [_ a b y]
    (vector-math MKL/vsCopySign ^RealBlockVector a ^RealBlockVector b ^RealBlockVector y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (vector-ramp this a y))
  (relu [this alpha a y]
    (vector-relu this alpha a y))
  (elu [this alpha a y]
    (vector-elu this alpha a y))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper x]
    (vector-random MKL/vsRngUniform (or rng-stream default-rng-stream)
                   lower upper ^RealBlockVector x))
  (rand-normal [_ rng-stream mu sigma x]
    (vector-random MKL/vsRngGaussian (or rng-stream default-rng-stream)
                   mu sigma ^RealBlockVector x)))

;; ================= General Matrix Engines ====================================

(def ^:private zero-matrix ^RealGEMatrix
  (->RealGEMatrix nil (full-storage true 0 0 Integer/MAX_VALUE) nil nil nil nil true
                  (ByteBuffer/allocateDirect 0) 0 0 0))

(deftype DoubleGEEngine []
  Blas
  (swap [_ a b]
    (matrix-swap CBLAS/dswap ^RealGEMatrix a ^RealGEMatrix b))
  (copy [_ a b]
    (ge-copy MKL/domatcopy ^RealGEMatrix a ^RealGEMatrix b))
  (scal [_ alpha a]
    (ge-scal MKL/dimatcopy alpha ^RealGEMatrix a))
  (dot [_ a b]
    (matrix-dot CBLAS/ddot ^RealGEMatrix a ^RealGEMatrix b))
  (nrm1 [_ a]
    (ge-lan LAPACK/dlange (int \O) ^RealGEMatrix a))
  (nrm2 [_ a]
    (ge-lan LAPACK/dlange (int \F) ^RealGEMatrix a))
  (nrmi [_ a]
    (ge-lan LAPACK/dlange (int \I) ^RealGEMatrix a))
  (asum [_ a]
    (matrix-sum CBLAS/dasum ^RealGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpby MKL/domatadd alpha ^RealGEMatrix a 1.0 ^RealGEMatrix b))
  (mv [_ alpha a x beta y]
    (ge-mv CBLAS/dgemv alpha ^RealGEMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk CBLAS/dger alpha ^RealBlockVector x ^RealBlockVector y ^RealGEMatrix a))
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm CBLAS/dgemm alpha ^RealGEMatrix a ^RealGEMatrix b beta ^RealGEMatrix c))
  BlasPlus
  (amax [_ a]
    (ge-lan LAPACK/dlange (int \M) ^RealGEMatrix a))
  (sum [_ a]
    (matrix-sum CBLAS/ddot ^RealGEMatrix a ^RealBlockVector ones-double))
  (set-all [_ alpha a]
    (ge-laset LAPACK/dlaset alpha alpha ^RealGEMatrix a))
  (axpby [_ alpha a beta b]
    (ge-axpby MKL/domatadd alpha ^RealGEMatrix a beta ^RealGEMatrix b))
  (trans [_ a]
    (ge-trans MKL/dimatcopy ^RealGEMatrix a))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealGEMatrix a increasing))
  (laswp [_ a ipiv k1 k2]
    (ge-laswp LAPACK/dlaswp ^RealGEMatrix a ^IntegerBlockVector ipiv k1 k2))
  (lapmr [_ a k forward]
    (ge-lapm LAPACK/dlapmr ^RealGEMatrix a ^IntegerBlockVector k ^Boolean forward))
  (lapmt [_ a k forward]
    (ge-lapm LAPACK/dlapmt ^RealGEMatrix a ^IntegerBlockVector k ^Boolean forward))
  (trf [_ a ipiv]
    (ge-trf LAPACK/dgetrf ^RealGEMatrix a ^IntegerBlockVector ipiv))
  (trf [_ _]
    (dragan-says-ex "Pivotless factorization is not available for general matrices."))
  (tri [_ lu ipiv]
    (ge-tri LAPACK/dgetri ^RealGEMatrix lu ^IntegerBlockVector ipiv))
  (trs [_ lu b ipiv]
    (ge-trs LAPACK/dgetrs ^RealGEMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (ge-sv LAPACK/dgesv ^RealGEMatrix a ^RealGEMatrix b pure))
  (con [_ lu _ nrm nrm1?]
    (ge-con LAPACK/dgecon ^RealGEMatrix lu nrm nrm1?))
  (qrf [_ a tau]
    (ge-lqrf LAPACK/dgeqrf ^RealGEMatrix a ^RealBlockVector tau))
  (qrfp [_ a tau]
    (ge-lqrf LAPACK/dgeqrfp ^RealGEMatrix a ^RealBlockVector tau))
  (qp3 [_ a jpiv tau]
    (ge-qp3 LAPACK/dgeqp3 ^RealGEMatrix a ^IntegerBlockVector jpiv ^RealBlockVector tau))
  (gqr [_ a tau]
    (or-glqr LAPACK/dorgqr ^RealGEMatrix a ^RealBlockVector tau))
  (mqr [_ a tau c left]
    (or-mlqr LAPACK/dormqr ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (rqf [_ a tau]
    (ge-lqrf LAPACK/dgerqf ^RealGEMatrix a ^RealBlockVector tau))
  (grq [_ a tau]
    (or-glqr LAPACK/dorgrq ^RealGEMatrix a ^RealBlockVector tau))
  (mrq [_ a tau c left]
    (or-mlqr LAPACK/dormrq ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (lqf [_ a tau]
    (ge-lqrf LAPACK/dgelqf ^RealGEMatrix a ^RealBlockVector tau))
  (glq [_ a tau]
    (or-glqr LAPACK/dorglq ^RealGEMatrix a ^RealBlockVector tau))
  (mlq [_ a tau c left]
    (or-mlqr LAPACK/dormlq ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (qlf [_ a tau]
    (ge-lqrf LAPACK/dgeqlf ^RealGEMatrix a ^RealBlockVector tau))
  (gql [_ a tau]
    (or-glqr LAPACK/dorgql ^RealGEMatrix a ^RealBlockVector tau))
  (mql [_ a tau c left]
    (or-mlqr LAPACK/dormql ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (ls [_ a b]
    (ge-ls LAPACK/dgels ^RealGEMatrix a ^RealGEMatrix b))
  (lse [_ a b c d x]
    (ge-lse LAPACK/dgglse ^RealGEMatrix a ^RealGEMatrix b
            ^RealBlockVector c ^RealBlockVector d ^RealBlockVector x))
  (gls [_ a b d x y]
    (ge-gls LAPACK/dggglm ^RealGEMatrix a ^RealGEMatrix b
            ^RealBlockVector d ^RealBlockVector x ^RealBlockVector y))
  (ev [_ a w vl vr]
    (let [vl (or vl zero-matrix)
          vr (or vr zero-matrix)]
      (ge-ev LAPACK/dgeev ^RealGEMatrix a ^RealGEMatrix w ^RealGEMatrix vl ^RealGEMatrix vr)))
  (es [_ a w vs]
    (let [vs (or vs zero-matrix)]
      (ge-es LAPACK/dgees ^RealGEMatrix a ^RealGEMatrix w ^RealGEMatrix vs)))
  (svd [_ a sigma u vt superb]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-svd LAPACK/dgesvd ^RealGEMatrix a ^RealDiagonalMatrix sigma
              ^RealGEMatrix u ^RealGEMatrix vt ^RealDiagonalMatrix superb)))
  (svd [_ a sigma superb]
    (ge-svd LAPACK/dgesvd ^RealGEMatrix a ^RealDiagonalMatrix sigma
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix ^RealDiagonalMatrix superb))
  (sdd [_ a sigma u vt]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-sdd LAPACK/dgesdd ^RealGEMatrix a ^RealDiagonalMatrix sigma ^RealGEMatrix u ^RealGEMatrix vt)))
  (sdd [_ a sigma]
    (ge-sdd LAPACK/dgesdd ^RealGEMatrix a ^RealDiagonalMatrix sigma
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix))
  VectorMath
  (sqr [_ a y]
    (matrix-math MKL/vdSqr ^RealGEMatrix a ^RealGEMatrix y))
  (mul [_ a b y]
    (matrix-math MKL/vdMul ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (div [_ a b y]
    (matrix-math MKL/vdDiv ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (inv [_ a y]
    (matrix-math MKL/vdInv ^RealGEMatrix a ^RealGEMatrix y))
  (abs [_ a y]
    (matrix-math MKL/vdAbs ^RealGEMatrix a ^RealGEMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (matrix-linear-frac MKL/vdLinearFrac ^RealGEMatrix a ^RealGEMatrix b
                        scalea shifta scaleb shiftb ^RealGEMatrix y))
  (fmod [_ a b y]
    (matrix-math MKL/vdFmod ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (frem [_ a b y]
    (matrix-math MKL/vdRemainder ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (sqrt [_ a y]
    (matrix-math MKL/vdSqrt ^RealGEMatrix a ^RealGEMatrix y))
  (inv-sqrt [_ a y]
    (matrix-math MKL/vdInvSqrt ^RealGEMatrix a ^RealGEMatrix y))
  (cbrt [_ a y]
    (matrix-math MKL/vdCbrt ^RealGEMatrix a ^RealGEMatrix y))
  (inv-cbrt [_ a y]
    (matrix-math MKL/vdInvCbrt ^RealGEMatrix a ^RealGEMatrix y))
  (pow2o3 [_ a y]
    (matrix-math MKL/vdPow2o3 ^RealGEMatrix a ^RealGEMatrix y))
  (pow3o2 [_ a y]
    (matrix-math MKL/vdPow3o2 ^RealGEMatrix a ^RealGEMatrix y))
  (pow [_ a b y]
    (matrix-math MKL/vdPow ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (powx [_ a b y]
    (matrix-powx MKL/vdPowx ^RealGEMatrix a b ^RealGEMatrix y))
  (hypot [_ a b y]
    (matrix-math MKL/vdHypot ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (exp [_ a y]
    (matrix-math MKL/vdExp ^RealGEMatrix a ^RealGEMatrix y))
  (expm1 [_ a y]
    (matrix-math MKL/vdExpm1 ^RealGEMatrix a ^RealGEMatrix y))
  (log [_ a y]
    (matrix-math MKL/vdLn ^RealGEMatrix a ^RealGEMatrix y))
  (log10 [_ a y]
    (matrix-math MKL/vdLog10 ^RealGEMatrix a ^RealGEMatrix y))
  (sin [_ a y]
    (matrix-math MKL/vdSin ^RealGEMatrix a ^RealGEMatrix y))
  (cos [_ a y]
    (matrix-math MKL/vdCos ^RealGEMatrix a ^RealGEMatrix y))
  (tan [_ a y]
    (matrix-math MKL/vdTan ^RealGEMatrix a ^RealGEMatrix y))
  (sincos [_ a y z]
    (matrix-math MKL/vdSinCos ^RealGEMatrix a ^RealGEMatrix y ^RealGEMatrix z))
  (asin [_ a y]
    (matrix-math MKL/vdAsin ^RealGEMatrix a ^RealGEMatrix y))
  (acos [_ a y]
    (matrix-math MKL/vdAcos ^RealGEMatrix a ^RealGEMatrix y))
  (atan [_ a y]
    (matrix-math MKL/vdAtan ^RealGEMatrix a ^RealGEMatrix y))
  (atan2 [_ a b y]
    (matrix-math MKL/vdAtan2 ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (sinh [_ a y]
    (matrix-math MKL/vdSinh ^RealGEMatrix a ^RealGEMatrix y))
  (cosh [_ a y]
    (matrix-math MKL/vdCosh ^RealGEMatrix a ^RealGEMatrix y))
  (tanh [_ a y]
    (matrix-math MKL/vdTanh ^RealGEMatrix a ^RealGEMatrix y))
  (asinh [_ a y]
    (matrix-math MKL/vdAsinh ^RealGEMatrix a ^RealGEMatrix y))
  (acosh [_ a y]
    (matrix-math MKL/vdAcosh ^RealGEMatrix a ^RealGEMatrix y))
  (atanh [_ a y]
    (matrix-math MKL/vdAtanh ^RealGEMatrix a ^RealGEMatrix y))
  (erf [_ a y]
    (matrix-math MKL/vdErf ^RealGEMatrix a ^RealGEMatrix y))
  (erfc [_ a y]
    (matrix-math MKL/vdErfc ^RealGEMatrix a ^RealGEMatrix y))
  (erf-inv [_ a y]
    (matrix-math MKL/vdErfInv ^RealGEMatrix a ^RealGEMatrix y))
  (erfc-inv [_ a y]
    (matrix-math MKL/vdErfcInv ^RealGEMatrix a ^RealGEMatrix y))
  (cdf-norm [_ a y]
    (matrix-math MKL/vdCdfNorm ^RealGEMatrix a ^RealGEMatrix y))
  (cdf-norm-inv [_ a y]
    (matrix-math MKL/vdCdfNormInv ^RealGEMatrix a ^RealGEMatrix y))
  (gamma [_ a y]
    (matrix-math MKL/vdGamma ^RealGEMatrix a ^RealGEMatrix y))
  (lgamma [_ a y]
    (matrix-math MKL/vdLGamma ^RealGEMatrix a ^RealGEMatrix y))
  (expint1 [_ a y]
    (matrix-math MKL/vdExpInt1 ^RealGEMatrix a ^RealGEMatrix y))
  (floor [_ a y]
    (matrix-math MKL/vdFloor ^RealGEMatrix a ^RealGEMatrix y))
  (fceil [_ a y]
    (matrix-math MKL/vdCeil ^RealGEMatrix a ^RealGEMatrix y))
  (trunc [_ a y]
    (matrix-math MKL/vdTrunc ^RealGEMatrix a ^RealGEMatrix y))
  (round [_ a y]
    (matrix-math MKL/vdRound ^RealGEMatrix a ^RealGEMatrix y))
  (modf [_ a y z]
    (matrix-math MKL/vdModf ^RealGEMatrix a ^RealGEMatrix y ^RealGEMatrix z))
  (frac [_ a y]
    (matrix-math MKL/vdFrac ^RealGEMatrix a ^RealGEMatrix y))
  (fmin [_ a b y]
    (matrix-math MKL/vdFmin ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (fmax [_ a b y]
    (matrix-math MKL/vdFmax ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (copy-sign [_ a b y]
    (matrix-math MKL/vdCopySign ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper a]
    (matrix-random MKL/vdRngUniform (or rng-stream default-rng-stream)
                   lower upper ^RealGEMatrix a))
  (rand-normal [_ rng-stream mu sigma a]
    (matrix-random MKL/vdRngGaussian (or rng-stream default-rng-stream)
                   mu sigma ^RealGEMatrix a)))

(deftype FloatGEEngine []
  Blas
  (swap [_ a b]
    (matrix-swap CBLAS/sswap ^RealGEMatrix a ^RealGEMatrix b))
  (copy [_ a b]
    (ge-copy MKL/somatcopy ^RealGEMatrix a ^RealGEMatrix b))
  (scal [_ alpha a]
    (ge-scal MKL/simatcopy alpha ^RealGEMatrix a))
  (dot [_ a b]
    (matrix-dot CBLAS/sdot ^RealGEMatrix a ^RealGEMatrix b))
  (nrm1 [_ a]
    (ge-lan LAPACK/slange (int \O) ^RealGEMatrix a))
  (nrm2 [_ a]
    (ge-lan LAPACK/slange (int \F) ^RealGEMatrix a))
  (nrmi [_ a]
    (ge-lan LAPACK/slange (int \I) ^RealGEMatrix a))
  (asum [_ a]
    (matrix-sum CBLAS/sasum ^RealGEMatrix a))
  (axpy [_ alpha a b]
    (ge-axpby MKL/somatadd alpha ^RealGEMatrix a 1.0 ^RealGEMatrix b))
  (mv [_ alpha a x beta y]
    (ge-mv CBLAS/sgemv alpha ^RealGEMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (ge-mv a))
  (rk [_ alpha x y a]
    (ge-rk CBLAS/sger alpha ^RealBlockVector x ^RealBlockVector y ^RealGEMatrix a))
  (mm [_ alpha a b _]
    (ge-mm alpha a b))
  (mm [_ alpha a b beta c _]
    (ge-mm CBLAS/sgemm alpha ^RealGEMatrix a ^RealGEMatrix b beta ^RealGEMatrix c))
  BlasPlus
  (amax [_ a]
    (ge-lan LAPACK/slange (int \M) ^RealGEMatrix a))
  (sum [_ a]
    (matrix-sum CBLAS/sdot ^RealGEMatrix a ^RealBlockVector ones-float))
  (set-all [_ alpha a]
    (ge-laset LAPACK/slaset alpha alpha ^RealGEMatrix a))
  (axpby [_ alpha a beta b]
    (ge-axpby MKL/somatadd alpha ^RealGEMatrix a beta ^RealGEMatrix b))
  (trans [_ a]
    (ge-trans MKL/simatcopy ^RealGEMatrix a))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealGEMatrix a increasing))
  (laswp [_ a ipiv k1 k2]
    (ge-laswp LAPACK/slaswp ^RealGEMatrix a ^IntegerBlockVector ipiv k1 k2))
  (lapmr [_ a k forward]
    (ge-lapm LAPACK/slapmr ^RealGEMatrix a ^IntegerBlockVector k ^Boolean forward))
  (lapmt [_ a k forward]
    (ge-lapm LAPACK/slapmt ^RealGEMatrix a ^IntegerBlockVector k ^Boolean forward))
  (trf [_ a ipiv]
    (ge-trf LAPACK/sgetrf ^RealGEMatrix a ^IntegerBlockVector ipiv))
  (trf [_ _]
    (dragan-says-ex "Pivotless factorization is not available for general matrices."))
  (tri [_ lu ipiv]
    (ge-tri LAPACK/sgetri ^RealGEMatrix lu ^IntegerBlockVector ipiv))
  (trs [_ lu b ipiv]
    (ge-trs LAPACK/sgetrs ^RealGEMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (ge-sv LAPACK/sgesv ^RealGEMatrix a ^RealGEMatrix b pure))
  (con [_ lu _ nrm nrm1?]
    (ge-con LAPACK/sgecon ^RealGEMatrix lu nrm nrm1?))
  (qrf [_ a tau]
    (ge-lqrf LAPACK/sgeqrf ^RealGEMatrix a ^RealBlockVector tau))
  (qrfp [_ a tau]
    (ge-lqrf LAPACK/sgeqrfp ^RealGEMatrix a ^RealBlockVector tau))
  (qp3 [_ a jpiv tau]
    (ge-qp3 LAPACK/sgeqp3 ^RealGEMatrix a ^IntegerBlockVector jpiv ^RealBlockVector tau))
  (gqr [_ a tau]
    (or-glqr LAPACK/sorgqr ^RealGEMatrix a ^RealBlockVector tau))
  (mqr [_ a tau c left]
    (or-mlqr LAPACK/sormqr ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (rqf [_ a tau]
    (ge-lqrf LAPACK/sgerqf ^RealGEMatrix a ^RealBlockVector tau))
  (grq [_ a tau]
    (or-glqr LAPACK/sorgrq ^RealGEMatrix a ^RealBlockVector tau))
  (mrq [_ a tau c left]
    (or-mlqr LAPACK/sormrq ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (lqf [_ a tau]
    (ge-lqrf LAPACK/sgelqf ^RealGEMatrix a ^RealBlockVector tau))
  (glq [_ a tau]
    (or-glqr LAPACK/sorglq ^RealGEMatrix a ^RealBlockVector tau))
  (mlq [_ a tau c left]
    (or-mlqr LAPACK/sormlq ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (qlf [_ a tau]
    (ge-lqrf LAPACK/sgeqlf ^RealGEMatrix a ^RealBlockVector tau))
  (gql [_ a tau]
    (or-glqr LAPACK/sorgql ^RealGEMatrix a ^RealBlockVector tau))
  (mql [_ a tau c left]
    (or-mlqr LAPACK/sormql ^RealGEMatrix a ^RealBlockVector tau ^RealGEMatrix c left))
  (ls [_ a b]
    (ge-ls LAPACK/sgels ^RealGEMatrix a ^RealGEMatrix b))
  (lse [_ a b c d x]
    (ge-lse LAPACK/sgglse ^RealGEMatrix a ^RealGEMatrix b
            ^RealBlockVector c ^RealBlockVector d ^RealBlockVector x))
  (gls [_ a b d x y]
    (ge-gls LAPACK/sggglm ^RealGEMatrix a ^RealGEMatrix b
            ^RealBlockVector d ^RealBlockVector x ^RealBlockVector y))
  (ev [_ a w vl vr]
    (let [vl (or vl zero-matrix)
          vr (or vr zero-matrix)]
      (ge-ev LAPACK/sgeev ^RealGEMatrix a ^RealGEMatrix w ^RealGEMatrix vl ^RealGEMatrix vr)))
  (es [_ a w vs]
    (let [vs (or vs zero-matrix)]
      (ge-es LAPACK/sgees ^RealGEMatrix a ^RealGEMatrix w ^RealGEMatrix vs)))
  (svd [_ a sigma u vt superb]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-svd LAPACK/sgesvd ^RealGEMatrix a ^RealDiagonalMatrix sigma
              ^RealGEMatrix u ^RealGEMatrix vt ^RealDiagonalMatrix superb)))
  (svd [_ a sigma superb]
    (ge-svd LAPACK/sgesvd ^RealGEMatrix a ^RealDiagonalMatrix sigma
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix ^RealDiagonalMatrix superb))
  (sdd [_ a sigma u vt]
    (let [u (or u zero-matrix)
          vt (or vt zero-matrix)]
      (ge-sdd LAPACK/sgesdd ^RealGEMatrix a ^RealDiagonalMatrix sigma ^RealGEMatrix u ^RealGEMatrix vt)))
  (sdd [_ a sigma]
    (ge-sdd LAPACK/sgesdd ^RealGEMatrix a ^RealDiagonalMatrix sigma
            ^RealGEMatrix zero-matrix ^RealGEMatrix zero-matrix))
  VectorMath
  (sqr [_ a y]
    (matrix-math MKL/vsSqr ^RealGEMatrix a ^RealGEMatrix y))
  (mul [_ a b y]
    (matrix-math MKL/vsMul ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (div [_ a b y]
    (matrix-math MKL/vsDiv ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (inv [_ a y]
    (matrix-math MKL/vsInv ^RealGEMatrix a ^RealGEMatrix y))
  (abs [_ a y]
    (matrix-math MKL/vsAbs ^RealGEMatrix a ^RealGEMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (matrix-linear-frac MKL/vsLinearFrac ^RealGEMatrix a ^RealGEMatrix b
                        scalea shifta scaleb shiftb ^RealGEMatrix y))
  (fmod [_ a b y]
    (matrix-math MKL/vsFmod ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (frem [_ a b y]
    (matrix-math MKL/vsRemainder ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (sqrt [_ a y]
    (matrix-math MKL/vsSqrt ^RealGEMatrix a ^RealGEMatrix y))
  (inv-sqrt [_ a y]
    (matrix-math MKL/vsInvSqrt ^RealGEMatrix a ^RealGEMatrix y))
  (cbrt [_ a y]
    (matrix-math MKL/vsCbrt ^RealGEMatrix a ^RealGEMatrix y))
  (inv-cbrt [_ a y]
    (matrix-math MKL/vsInvCbrt ^RealGEMatrix a ^RealGEMatrix y))
  (pow2o3 [_ a y]
    (matrix-math MKL/vsPow2o3 ^RealGEMatrix a ^RealGEMatrix y))
  (pow3o2 [_ a y]
    (matrix-math MKL/vsPow3o2 ^RealGEMatrix a ^RealGEMatrix y))
  (pow [_ a b y]
    (matrix-math MKL/vsPow ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (powx [_ a b y]
    (matrix-powx MKL/vsPowx ^RealGEMatrix a b ^RealGEMatrix y))
  (hypot [_ a b y]
    (matrix-math MKL/vsHypot ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (exp [_ a y]
    (matrix-math MKL/vsExp ^RealGEMatrix a ^RealGEMatrix y))
  (expm1 [_ a y]
    (matrix-math MKL/vsExpm1 ^RealGEMatrix a ^RealGEMatrix y))
  (log [_ a y]
    (matrix-math MKL/vsLn ^RealGEMatrix a ^RealGEMatrix y))
  (log10 [_ a y]
    (matrix-math MKL/vsLog10 ^RealGEMatrix a ^RealGEMatrix y))
  (sin [_ a y]
    (matrix-math MKL/vsSin ^RealGEMatrix a ^RealGEMatrix y))
  (cos [_ a y]
    (matrix-math MKL/vsCos ^RealGEMatrix a ^RealGEMatrix y))
  (tan [_ a y]
    (matrix-math MKL/vsTan ^RealGEMatrix a ^RealGEMatrix y))
  (sincos [_ a y z]
    (matrix-math MKL/vsSinCos ^RealGEMatrix a ^RealGEMatrix y ^RealGEMatrix z))
  (asin [_ a y]
    (matrix-math MKL/vsAsin ^RealGEMatrix a ^RealGEMatrix y))
  (acos [_ a y]
    (matrix-math MKL/vsAcos ^RealGEMatrix a ^RealGEMatrix y))
  (atan [_ a y]
    (matrix-math MKL/vsAtan ^RealGEMatrix a ^RealGEMatrix y))
  (atan2 [_ a b y]
    (matrix-math MKL/vsAtan2 ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (sinh [_ a y]
    (matrix-math MKL/vsSinh ^RealGEMatrix a ^RealGEMatrix y))
  (cosh [_ a y]
    (matrix-math MKL/vsCosh ^RealGEMatrix a ^RealGEMatrix y))
  (tanh [_ a y]
    (matrix-math MKL/vsTanh ^RealGEMatrix a ^RealGEMatrix y))
  (asinh [_ a y]
    (matrix-math MKL/vsAsinh ^RealGEMatrix a ^RealGEMatrix y))
  (acosh [_ a y]
    (matrix-math MKL/vsAcosh ^RealGEMatrix a ^RealGEMatrix y))
  (atanh [_ a y]
    (matrix-math MKL/vsAtanh ^RealGEMatrix a ^RealGEMatrix y))
  (erf [_ a y]
    (matrix-math MKL/vsErf ^RealGEMatrix a ^RealGEMatrix y))
  (erfc [_ a y]
    (matrix-math MKL/vsErfc ^RealGEMatrix a ^RealGEMatrix y))
  (erf-inv [_ a y]
    (matrix-math MKL/vsErfInv ^RealGEMatrix a ^RealGEMatrix y))
  (erfc-inv [_ a y]
    (matrix-math MKL/vsErfcInv ^RealGEMatrix a ^RealGEMatrix y))
  (cdf-norm [_ a y]
    (matrix-math MKL/vsCdfNorm ^RealGEMatrix a ^RealGEMatrix y))
  (cdf-norm-inv [_ a y]
    (matrix-math MKL/vsCdfNormInv ^RealGEMatrix a ^RealGEMatrix y))
  (gamma [_ a y]
    (matrix-math MKL/vsGamma ^RealGEMatrix a ^RealGEMatrix y))
  (lgamma [_ a y]
    (matrix-math MKL/vsLGamma ^RealGEMatrix a ^RealGEMatrix y))
  (expint1 [_ a y]
    (matrix-math MKL/vsExpInt1 ^RealGEMatrix a ^RealGEMatrix y))
  (floor [_ a y]
    (matrix-math MKL/vsFloor ^RealGEMatrix a ^RealGEMatrix y))
  (fceil [_ a y]
    (matrix-math MKL/vsCeil ^RealGEMatrix a ^RealGEMatrix y))
  (trunc [_ a y]
    (matrix-math MKL/vsTrunc ^RealGEMatrix a ^RealGEMatrix y))
  (round [_ a y]
    (matrix-math MKL/vsRound ^RealGEMatrix a ^RealGEMatrix y))
  (modf [_ a y z]
    (matrix-math MKL/vsModf ^RealGEMatrix a ^RealGEMatrix y ^RealGEMatrix z))
  (frac [_ a y]
    (matrix-math MKL/vsFrac ^RealGEMatrix a ^RealGEMatrix y))
  (fmin [_ a b y]
    (matrix-math MKL/vsFmin ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (fmax [_ a b y]
    (matrix-math MKL/vsFmax ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (copy-sign [_ a b y]
    (matrix-math MKL/vsCopySign ^RealGEMatrix a ^RealGEMatrix b ^RealGEMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y))
  RandomNumberGenerator
  (rand-uniform [_ rng-stream lower upper a]
    (matrix-random MKL/vsRngUniform (or rng-stream default-rng-stream)
                   lower upper ^RealGEMatrix a))
  (rand-normal [_ rng-stream mu sigma a]
    (matrix-random MKL/vsRngGaussian (or rng-stream default-rng-stream)
                   mu sigma ^RealGEMatrix a)))

;; ================= Triangular Matrix Engines =================================

(deftype DoubleTREngine []
  Blas
  (swap [_ a b]
    (matrix-swap CBLAS/dswap ^RealUploMatrix a ^RealUploMatrix b))
  (copy [_ a b]
    (tr-lacpy LAPACK/dlacpy CBLAS/dcopy ^RealUploMatrix a ^RealUploMatrix b))
  (scal [_ alpha a]
    (tr-lascl LAPACK/dlascl alpha ^RealUploMatrix a))
  (dot [_ a b]
    (tr-dot CBLAS/ddot ^RealUploMatrix a ^RealUploMatrix b))
  (nrm1 [_ a]
    (tr-lan LAPACK/dlantr (int \O) ^RealUploMatrix a))
  (nrm2 [_ a]
    (tr-lan LAPACK/dlantr (int \F) ^RealUploMatrix a))
  (nrmi [_ a]
    (tr-lan LAPACK/dlantr (int \I) ^RealUploMatrix a))
  (asum [_ a]
    (tr-sum CBLAS/dasum ^RealUploMatrix a))
  (axpy [_ alpha a b]
    (matrix-axpy CBLAS/daxpy alpha ^RealUploMatrix a ^RealUploMatrix b))
  (mv [_ _ a _ _ _]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv CBLAS/dtrmv ^RealUploMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm CBLAS/dtrmm alpha ^RealUploMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tr-lan LAPACK/dlantr (int \M) ^RealUploMatrix a))
  (sum [_ a]
    (tr-sum CBLAS/dsum ^RealUploMatrix a))
  (set-all [_ alpha a]
    (tr-laset LAPACK/dlaset alpha alpha ^RealUploMatrix a) a)
  (axpby [_ alpha a beta b]
    (matrix-axpby MKL/daxpby alpha ^RealUploMatrix a beta ^RealUploMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for TR matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealUploMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TR matrices."))
  (tri [_ a]
    (tr-tri LAPACK/dtrtri ^RealUploMatrix a))
  (trs [_ a b]
    (tr-trs LAPACK/dtrtrs ^RealUploMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tr-sv CBLAS/dtrsm ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tr-con LAPACK/dtrcon ^RealUploMatrix a nrm1?))
  VectorMath
  (sqr [_ a y]
    (matrix-math MKL/vdSqr ^RealUploMatrix a ^RealUploMatrix y))
  (mul [_ a b y]
    (matrix-math MKL/vdMul ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (div [_ a b y]
    (matrix-math MKL/vdDiv ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (inv [_ a y]
    (matrix-math MKL/vdInv ^RealUploMatrix a ^RealUploMatrix y))
  (abs [_ a y]
    (matrix-math MKL/vdAbs ^RealUploMatrix a ^RealUploMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (matrix-linear-frac MKL/vdLinearFrac ^RealUploMatrix a ^RealUploMatrix b
                        scalea shifta scaleb shiftb ^RealUploMatrix y))
  (fmod [_ a b y]
    (matrix-math MKL/vdFmod ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (frem [_ a b y]
    (matrix-math MKL/vdRemainder ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sqrt [_ a y]
    (matrix-math MKL/vdSqrt ^RealUploMatrix a ^RealUploMatrix y))
  (inv-sqrt [_ a y]
    (matrix-math MKL/vdInvSqrt ^RealUploMatrix a ^RealUploMatrix y))
  (cbrt [_ a y]
    (matrix-math MKL/vdCbrt ^RealUploMatrix a ^RealUploMatrix y))
  (inv-cbrt [_ a y]
    (matrix-math MKL/vdInvCbrt ^RealUploMatrix a ^RealUploMatrix y))
  (pow2o3 [_ a y]
    (matrix-math MKL/vdPow2o3 ^RealUploMatrix a ^RealUploMatrix y))
  (pow3o2 [_ a y]
    (matrix-math MKL/vdPow3o2 ^RealUploMatrix a ^RealUploMatrix y))
  (pow [_ a b y]
    (matrix-math MKL/vdPow ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (powx [_ a b y]
    (matrix-powx MKL/vdPowx ^RealUploMatrix a b ^RealUploMatrix y))
  (hypot [_ a b y]
    (matrix-math MKL/vdHypot ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (exp [_ a y]
    (matrix-math MKL/vdExp ^RealUploMatrix a ^RealUploMatrix y))
  (expm1 [_ a y]
    (matrix-math MKL/vdExpm1 ^RealUploMatrix a ^RealUploMatrix y))
  (log [_ a y]
    (matrix-math MKL/vdLn ^RealUploMatrix a ^RealUploMatrix y))
  (log10 [_ a y]
    (matrix-math MKL/vdLog10 ^RealUploMatrix a ^RealUploMatrix y))
  (sin [_ a y]
    (matrix-math MKL/vdSin ^RealUploMatrix a ^RealUploMatrix y))
  (cos [_ a y]
    (matrix-math MKL/vdCos ^RealUploMatrix a ^RealUploMatrix y))
  (tan [_ a y]
    (matrix-math MKL/vdTan ^RealUploMatrix a ^RealUploMatrix y))
  (sincos [_ a y z]
    (matrix-math MKL/vdSinCos ^RealUploMatrix a ^RealUploMatrix y ^RealUploMatrix z))
  (asin [_ a y]
    (matrix-math MKL/vdAsin ^RealUploMatrix a ^RealUploMatrix y))
  (acos [_ a y]
    (matrix-math MKL/vdAcos ^RealUploMatrix a ^RealUploMatrix y))
  (atan [_ a y]
    (matrix-math MKL/vdAtan ^RealUploMatrix a ^RealUploMatrix y))
  (atan2 [_ a b y]
    (matrix-math MKL/vdAtan2 ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sinh [_ a y]
    (matrix-math MKL/vdSinh ^RealUploMatrix a ^RealUploMatrix y))
  (cosh [_ a y]
    (matrix-math MKL/vdCosh ^RealUploMatrix a ^RealUploMatrix y))
  (tanh [_ a y]
    (matrix-math MKL/vdTanh ^RealUploMatrix a ^RealUploMatrix y))
  (asinh [_ a y]
    (matrix-math MKL/vdAsinh ^RealUploMatrix a ^RealUploMatrix y))
  (acosh [_ a y]
    (matrix-math MKL/vdAcosh ^RealUploMatrix a ^RealUploMatrix y))
  (atanh [_ a y]
    (matrix-math MKL/vdAtanh ^RealUploMatrix a ^RealUploMatrix y))
  (erf [_ a y]
    (matrix-math MKL/vdErf ^RealUploMatrix a ^RealUploMatrix y))
  (erfc [_ a y]
    (matrix-math MKL/vdErfc ^RealUploMatrix a ^RealUploMatrix y))
  (erf-inv [_ a y]
    (matrix-math MKL/vdErfInv ^RealUploMatrix a ^RealUploMatrix y))
  (erfc-inv [_ a y]
    (matrix-math MKL/vdErfcInv ^RealUploMatrix a ^RealUploMatrix y))
  (cdf-norm [_ a y]
    (matrix-math MKL/vdCdfNorm ^RealUploMatrix a ^RealUploMatrix y))
  (cdf-norm-inv [_ a y]
    (matrix-math MKL/vdCdfNormInv ^RealUploMatrix a ^RealUploMatrix y))
  (gamma [_ a y]
    (matrix-math MKL/vdGamma ^RealUploMatrix a ^RealUploMatrix y))
  (lgamma [_ a y]
    (matrix-math MKL/vdLGamma ^RealUploMatrix a ^RealUploMatrix y))
  (expint1 [_ a y]
    (matrix-math MKL/vdExpInt1 ^RealUploMatrix a ^RealUploMatrix y))
  (floor [_ a y]
    (matrix-math MKL/vdFloor ^RealUploMatrix a ^RealUploMatrix y))
  (fceil [_ a y]
    (matrix-math MKL/vdCeil ^RealUploMatrix a ^RealUploMatrix y))
  (trunc [_ a y]
    (matrix-math MKL/vdTrunc ^RealUploMatrix a ^RealUploMatrix y))
  (round [_ a y]
    (matrix-math MKL/vdRound ^RealUploMatrix a ^RealUploMatrix y))
  (modf [_ a y z]
    (matrix-math MKL/vdModf ^RealUploMatrix a ^RealUploMatrix y ^RealUploMatrix z))
  (frac [_ a y]
    (matrix-math MKL/vdFrac ^RealUploMatrix a ^RealUploMatrix y))
  (fmin [_ a b y]
    (matrix-math MKL/vdFmin ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (fmax [_ a b y]
    (matrix-math MKL/vdFmax ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (copy-sign [_ a b y]
    (matrix-math MKL/vdCopySign ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype FloatTREngine []
  Blas
  (swap [_ a b]
    (matrix-swap CBLAS/sswap ^RealUploMatrix a ^RealUploMatrix b))
  (copy [_ a b]
    (tr-lacpy LAPACK/slacpy CBLAS/scopy ^RealUploMatrix a ^RealUploMatrix b))
  (scal [_ alpha a]
    (tr-lascl LAPACK/slascl alpha ^RealUploMatrix a))
  (dot [_ a b]
    (tr-dot CBLAS/sdot ^RealUploMatrix a ^RealUploMatrix b))
  (nrm1 [_ a]
    (tr-lan LAPACK/slantr (int \O) ^RealUploMatrix a))
  (nrm2 [_ a]
    (tr-lan LAPACK/slantr (int \F) ^RealUploMatrix a))
  (nrmi [_ a]
    (tr-lan LAPACK/slantr (int \I) ^RealUploMatrix a))
  (asum [_ a]
    (tr-sum CBLAS/sasum ^RealUploMatrix a))
  (axpy [_ alpha a b]
    (matrix-axpy CBLAS/saxpy alpha ^RealUploMatrix a ^RealUploMatrix b))
  (mv [_ _ a _ _ _]
    (tr-mv a))
  (mv [_ a x]
    (tr-mv CBLAS/strmv ^RealUploMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tr-mm a))
  (mm [_ alpha a b left]
    (tr-mm CBLAS/strmm alpha ^RealUploMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tr-lan LAPACK/slantr (int \M) ^RealUploMatrix a))
  (sum [_ a]
    (tr-sum  CBLAS/ssum ^RealUploMatrix a))
  (set-all [_ alpha a]
    (tr-laset LAPACK/slaset alpha alpha ^RealUploMatrix a))
  (axpby [_ alpha a beta b]
    (matrix-axpby MKL/saxpby alpha ^RealUploMatrix a beta ^RealUploMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for TR matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealUploMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TR matrices."))
  (tri [_ a]
    (tr-tri LAPACK/strtri ^RealUploMatrix a))
  (trs [_ a b]
    (tr-trs LAPACK/strtrs ^RealUploMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tr-sv CBLAS/strsm ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tr-con LAPACK/strcon ^RealUploMatrix a nrm1?))
  VectorMath
  (sqr [_ a y]
    (matrix-math MKL/vsSqr ^RealUploMatrix a ^RealUploMatrix y))
  (mul [_ a b y]
    (matrix-math MKL/vsMul ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (div [_ a b y]
    (matrix-math MKL/vsDiv ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (inv [_ a y]
    (matrix-math MKL/vsInv ^RealUploMatrix a ^RealUploMatrix y))
  (abs [_ a y]
    (matrix-math MKL/vsAbs ^RealUploMatrix a ^RealUploMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (matrix-linear-frac MKL/vsLinearFrac ^RealUploMatrix a ^RealUploMatrix b
                        scalea shifta scaleb shiftb ^RealUploMatrix y))
  (fmod [_ a b y]
    (matrix-math MKL/vsFmod ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (frem [_ a b y]
    (matrix-math MKL/vsRemainder ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sqrt [_ a y]
    (matrix-math MKL/vsSqrt ^RealUploMatrix a ^RealUploMatrix y))
  (inv-sqrt [_ a y]
    (matrix-math MKL/vsInvSqrt ^RealUploMatrix a ^RealUploMatrix y))
  (cbrt [_ a y]
    (matrix-math MKL/vsCbrt ^RealUploMatrix a ^RealUploMatrix y))
  (inv-cbrt [_ a y]
    (matrix-math MKL/vsInvCbrt ^RealUploMatrix a ^RealUploMatrix y))
  (pow2o3 [_ a y]
    (matrix-math MKL/vsPow2o3 ^RealUploMatrix a ^RealUploMatrix y))
  (pow3o2 [_ a y]
    (matrix-math MKL/vsPow3o2 ^RealUploMatrix a ^RealUploMatrix y))
  (pow [_ a b y]
    (matrix-math MKL/vsPow ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (powx [_ a b y]
    (matrix-powx MKL/vsPowx ^RealUploMatrix a b ^RealUploMatrix y))
  (hypot [_ a b y]
    (matrix-math MKL/vsHypot ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (exp [_ a y]
    (matrix-math MKL/vsExp ^RealUploMatrix a ^RealUploMatrix y))
  (expm1 [_ a y]
    (matrix-math MKL/vsExpm1 ^RealUploMatrix a ^RealUploMatrix y))
  (log [_ a y]
    (matrix-math MKL/vsLn ^RealUploMatrix a ^RealUploMatrix y))
  (log10 [_ a y]
    (matrix-math MKL/vsLog10 ^RealUploMatrix a ^RealUploMatrix y))
  (sin [_ a y]
    (matrix-math MKL/vsSin ^RealUploMatrix a ^RealUploMatrix y))
  (cos [_ a y]
    (matrix-math MKL/vsCos ^RealUploMatrix a ^RealUploMatrix y))
  (tan [_ a y]
    (matrix-math MKL/vsTan ^RealUploMatrix a ^RealUploMatrix y))
  (sincos [_ a y z]
    (matrix-math MKL/vsSinCos ^RealUploMatrix a ^RealUploMatrix y ^RealUploMatrix z))
  (asin [_ a y]
    (matrix-math MKL/vsAsin ^RealUploMatrix a ^RealUploMatrix y))
  (acos [_ a y]
    (matrix-math MKL/vsAcos ^RealUploMatrix a ^RealUploMatrix y))
  (atan [_ a y]
    (matrix-math MKL/vsAtan ^RealUploMatrix a ^RealUploMatrix y))
  (atan2 [_ a b y]
    (matrix-math MKL/vsAtan2 ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sinh [_ a y]
    (matrix-math MKL/vsSinh ^RealUploMatrix a ^RealUploMatrix y))
  (cosh [_ a y]
    (matrix-math MKL/vsCosh ^RealUploMatrix a ^RealUploMatrix y))
  (tanh [_ a y]
    (matrix-math MKL/vsTanh ^RealUploMatrix a ^RealUploMatrix y))
  (asinh [_ a y]
    (matrix-math MKL/vsAsinh ^RealUploMatrix a ^RealUploMatrix y))
  (acosh [_ a y]
    (matrix-math MKL/vsAcosh ^RealUploMatrix a ^RealUploMatrix y))
  (atanh [_ a y]
    (matrix-math MKL/vsAtanh ^RealUploMatrix a ^RealUploMatrix y))
  (erf [_ a y]
    (matrix-math MKL/vsErf ^RealUploMatrix a ^RealUploMatrix y))
  (erfc [_ a y]
    (matrix-math MKL/vsErfc ^RealUploMatrix a ^RealUploMatrix y))
  (erf-inv [_ a y]
    (matrix-math MKL/vsErfInv ^RealUploMatrix a ^RealUploMatrix y))
  (erfc-inv [_ a y]
    (matrix-math MKL/vsErfcInv ^RealUploMatrix a ^RealUploMatrix y))
  (cdf-norm [_ a y]
    (matrix-math MKL/vsCdfNorm ^RealUploMatrix a ^RealUploMatrix y))
  (cdf-norm-inv [_ a y]
    (matrix-math MKL/vsCdfNormInv ^RealUploMatrix a ^RealUploMatrix y))
  (gamma [_ a y]
    (matrix-math MKL/vsGamma ^RealUploMatrix a ^RealUploMatrix y))
  (lgamma [_ a y]
    (matrix-math MKL/vsLGamma ^RealUploMatrix a ^RealUploMatrix y))
  (expint1 [_ a y]
    (matrix-math MKL/vsExpInt1 ^RealUploMatrix a ^RealUploMatrix y))
  (floor [_ a y]
    (matrix-math MKL/vsFloor ^RealUploMatrix a ^RealUploMatrix y))
  (fceil [_ a y]
    (matrix-math MKL/vsCeil ^RealUploMatrix a ^RealUploMatrix y))
  (trunc [_ a y]
    (matrix-math MKL/vsTrunc ^RealUploMatrix a ^RealUploMatrix y))
  (round [_ a y]
    (matrix-math MKL/vsRound ^RealUploMatrix a ^RealUploMatrix y))
  (modf [_ a y z]
    (matrix-math MKL/vsModf ^RealUploMatrix a ^RealUploMatrix y ^RealUploMatrix z))
  (frac [_ a y]
    (matrix-math MKL/vsFrac ^RealUploMatrix a ^RealUploMatrix y))
  (fmin [_ a b y]
    (matrix-math MKL/vsFmin ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (fmax [_ a b y]
    (matrix-math MKL/vsFmax ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (copy-sign [_ a b y]
    (matrix-math MKL/vsCopySign ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

;; =============== Symmetric Matrix Engines ===================================

(deftype DoubleSYEngine []
  Blas
  (swap [_ a b]
    (sy-swap CBLAS/dswap ^RealUploMatrix a ^RealUploMatrix b))
  (copy [_ a b]
    (sy-lacpy LAPACK/dlacpy CBLAS/dcopy ^RealUploMatrix a ^RealUploMatrix b))
  (scal [_ alpha a]
    (sy-lascl LAPACK/dlascl alpha ^RealUploMatrix a))
  (dot [_ a b]
    (sy-dot CBLAS/ddot ^RealUploMatrix a ^RealUploMatrix b))
  (nrm1 [_ a]
    (sy-lan LAPACK/dlansy (int \O) ^RealUploMatrix a))
  (nrm2 [_ a]
    (sy-lan LAPACK/dlansy (int \F) ^RealUploMatrix a))
  (nrmi [_ a]
    (sy-lan LAPACK/dlansy (int \I) ^RealUploMatrix a))
  (asum [_ a]
    (sy-sum CBLAS/dasum ^RealUploMatrix a))
  (axpy [_ alpha a b]
    (sy-axpy CBLAS/daxpy alpha ^RealUploMatrix a ^RealUploMatrix b))
  (mv [_ alpha a x beta y]
    (sy-mv CBLAS/dsymv alpha ^RealUploMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sy-mv a))
  (rk [_ alpha x y a]
    (sy-r CBLAS/dsyr2 alpha ^RealBlockVector x ^RealBlockVector y ^RealUploMatrix a))
  (rk [_ alpha x a]
    (sy-r CBLAS/dsyr alpha ^RealBlockVector x ^RealUploMatrix a))
  (srk [_ alpha a beta c]
    (sy-rk CBLAS/dsyrk alpha ^RealGEMatrix a beta ^RealUploMatrix c))
  (mm [_ alpha a b beta c left]
    (sy-mm CBLAS/dsymm alpha ^RealUploMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ alpha a b _]
    (sy-mm a))
  BlasPlus
  (amax [_ a]
    (sy-lan LAPACK/dlansy (int \M) ^RealUploMatrix a))
  (sum [_ a]
    (sy-sum CBLAS/dsum ^RealUploMatrix a))
  (set-all [_ alpha a]
    (sy-laset LAPACK/dlaset alpha alpha ^RealUploMatrix a))
  (axpby [_ alpha a beta b]
    (sy-axpby MKL/daxpby alpha ^RealUploMatrix a beta ^RealUploMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for SY matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealUploMatrix a increasing))
  (trf [_ a ipiv]
    (sy-trx LAPACK/dsytrf ^RealUploMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (sy-trx LAPACK/dpotrf ^RealUploMatrix a))
  (trfx [_ a]
    (sy-trfx LAPACK/dpotrf ^RealUploMatrix a))
  (tri [_ ldl ipiv]
    (sy-trx LAPACK/dsytri ^RealUploMatrix ldl ^IntegerBlockVector ipiv))
  (tri [_ gg]
    (sy-trx LAPACK/dpotri ^RealUploMatrix gg))
  (trs [_ ldl b ipiv]
    (sy-trs LAPACK/dsytrs ^RealUploMatrix ldl ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (trs [_ gg b]
    (sy-trs LAPACK/dpotrs ^RealUploMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sy-sv LAPACK/dposv LAPACK/dsysv ^RealUploMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sy-sv LAPACK/dposv ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ ldl ipiv nrm _]
    (sy-con LAPACK/dsycon ^RealUploMatrix ldl ^IntegerBlockVector ipiv nrm))
  (con [_ gg nrm _]
    (sy-con LAPACK/dpocon ^RealUploMatrix gg nrm))
  (ev [_ a w vl vr]
    (let [v (or vl vr zero-matrix)]
      (sy-ev LAPACK/dsyevd LAPACK/dsyevr ^RealUploMatrix a ^RealGEMatrix w ^RealGEMatrix v)))
  VectorMath
  (sqr [_ a y]
    (matrix-math MKL/vdSqr ^RealUploMatrix a ^RealUploMatrix y))
  (mul [_ a b y]
    (matrix-math MKL/vdMul ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (div [_ a b y]
    (matrix-math MKL/vdDiv ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (inv [_ a y]
    (matrix-math MKL/vdInv ^RealUploMatrix a ^RealUploMatrix y))
  (abs [_ a y]
    (matrix-math MKL/vdAbs ^RealUploMatrix a ^RealUploMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (matrix-linear-frac MKL/vdLinearFrac ^RealUploMatrix a ^RealUploMatrix b
                        scalea shifta scaleb shiftb ^RealUploMatrix y))
  (fmod [_ a b y]
    (matrix-math MKL/vdFmod ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (frem [_ a b y]
    (matrix-math MKL/vdRemainder ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sqrt [_ a y]
    (matrix-math MKL/vdSqrt ^RealUploMatrix a ^RealUploMatrix y))
  (inv-sqrt [_ a y]
    (matrix-math MKL/vdInvSqrt ^RealUploMatrix a ^RealUploMatrix y))
  (cbrt [_ a y]
    (matrix-math MKL/vdCbrt ^RealUploMatrix a ^RealUploMatrix y))
  (inv-cbrt [_ a y]
    (matrix-math MKL/vdInvCbrt ^RealUploMatrix a ^RealUploMatrix y))
  (pow2o3 [_ a y]
    (matrix-math MKL/vdPow2o3 ^RealUploMatrix a ^RealUploMatrix y))
  (pow3o2 [_ a y]
    (matrix-math MKL/vdPow3o2 ^RealUploMatrix a ^RealUploMatrix y))
  (pow [_ a b y]
    (matrix-math MKL/vdPow ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (powx [_ a b y]
    (matrix-powx MKL/vdPowx ^RealUploMatrix a b ^RealUploMatrix y))
  (hypot [_ a b y]
    (matrix-math MKL/vdHypot ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (exp [_ a y]
    (matrix-math MKL/vdExp ^RealUploMatrix a ^RealUploMatrix y))
  (expm1 [_ a y]
    (matrix-math MKL/vdExpm1 ^RealUploMatrix a ^RealUploMatrix y))
  (log [_ a y]
    (matrix-math MKL/vdLn ^RealUploMatrix a ^RealUploMatrix y))
  (log10 [_ a y]
    (matrix-math MKL/vdLog10 ^RealUploMatrix a ^RealUploMatrix y))
  (sin [_ a y]
    (matrix-math MKL/vdSin ^RealUploMatrix a ^RealUploMatrix y))
  (cos [_ a y]
    (matrix-math MKL/vdCos ^RealUploMatrix a ^RealUploMatrix y))
  (tan [_ a y]
    (matrix-math MKL/vdTan ^RealUploMatrix a ^RealUploMatrix y))
  (sincos [_ a y z]
    (matrix-math MKL/vdSinCos ^RealUploMatrix a ^RealUploMatrix y ^RealUploMatrix z))
  (asin [_ a y]
    (matrix-math MKL/vdAsin ^RealUploMatrix a ^RealUploMatrix y))
  (acos [_ a y]
    (matrix-math MKL/vdAcos ^RealUploMatrix a ^RealUploMatrix y))
  (atan [_ a y]
    (matrix-math MKL/vdAtan ^RealUploMatrix a ^RealUploMatrix y))
  (atan2 [_ a b y]
    (matrix-math MKL/vdAtan2 ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sinh [_ a y]
    (matrix-math MKL/vdSinh ^RealUploMatrix a ^RealUploMatrix y))
  (cosh [_ a y]
    (matrix-math MKL/vdCosh ^RealUploMatrix a ^RealUploMatrix y))
  (tanh [_ a y]
    (matrix-math MKL/vdTanh ^RealUploMatrix a ^RealUploMatrix y))
  (asinh [_ a y]
    (matrix-math MKL/vdAsinh ^RealUploMatrix a ^RealUploMatrix y))
  (acosh [_ a y]
    (matrix-math MKL/vdAcosh ^RealUploMatrix a ^RealUploMatrix y))
  (atanh [_ a y]
    (matrix-math MKL/vdAtanh ^RealUploMatrix a ^RealUploMatrix y))
  (erf [_ a y]
    (matrix-math MKL/vdErf ^RealUploMatrix a ^RealUploMatrix y))
  (erfc [_ a y]
    (matrix-math MKL/vdErfc ^RealUploMatrix a ^RealUploMatrix y))
  (erf-inv [_ a y]
    (matrix-math MKL/vdErfInv ^RealUploMatrix a ^RealUploMatrix y))
  (erfc-inv [_ a y]
    (matrix-math MKL/vdErfcInv ^RealUploMatrix a ^RealUploMatrix y))
  (cdf-norm [_ a y]
    (matrix-math MKL/vdCdfNorm ^RealUploMatrix a ^RealUploMatrix y))
  (cdf-norm-inv [_ a y]
    (matrix-math MKL/vdCdfNormInv ^RealUploMatrix a ^RealUploMatrix y))
  (gamma [_ a y]
    (matrix-math MKL/vdGamma ^RealUploMatrix a ^RealUploMatrix y))
  (lgamma [_ a y]
    (matrix-math MKL/vdLGamma ^RealUploMatrix a ^RealUploMatrix y))
  (expint1 [_ a y]
    (matrix-math MKL/vdExpInt1 ^RealUploMatrix a ^RealUploMatrix y))
  (floor [_ a y]
    (matrix-math MKL/vdFloor ^RealUploMatrix a ^RealUploMatrix y))
  (fceil [_ a y]
    (matrix-math MKL/vdCeil ^RealUploMatrix a ^RealUploMatrix y))
  (trunc [_ a y]
    (matrix-math MKL/vdTrunc ^RealUploMatrix a ^RealUploMatrix y))
  (round [_ a y]
    (matrix-math MKL/vdRound ^RealUploMatrix a ^RealUploMatrix y))
  (modf [_ a y z]
    (matrix-math MKL/vdModf ^RealUploMatrix a ^RealUploMatrix y ^RealUploMatrix z))
  (frac [_ a y]
    (matrix-math MKL/vdFrac ^RealUploMatrix a ^RealUploMatrix y))
  (fmin [_ a b y]
    (matrix-math MKL/vdFmin ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (fmax [_ a b y]
    (matrix-math MKL/vdFmax ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (copy-sign [_ a b y]
    (matrix-math MKL/vdCopySign ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype FloatSYEngine []
  Blas
  (swap [_ a b]
    (sy-swap CBLAS/sswap ^RealUploMatrix a ^RealUploMatrix b))
  (copy [_ a b]
    (sy-lacpy LAPACK/slacpy CBLAS/scopy ^RealUploMatrix a ^RealUploMatrix b))
  (scal [_ alpha a]
    (sy-lascl LAPACK/slascl alpha ^RealUploMatrix a))
  (dot [_ a b]
    (sy-dot CBLAS/sdot ^RealUploMatrix a ^RealUploMatrix b))
  (nrm1 [_ a]
    (sy-lan LAPACK/slansy (int \O) ^RealUploMatrix a))
  (nrm2 [_ a]
    (sy-lan LAPACK/slansy (int \F) ^RealUploMatrix a))
  (nrmi [_ a]
    (sy-lan LAPACK/slansy (int \I) ^RealUploMatrix a))
  (asum [_ a]
    (sy-sum CBLAS/sasum ^RealUploMatrix a))
  (axpy [_ alpha a b]
    (sy-axpy CBLAS/saxpy alpha ^RealUploMatrix a ^RealUploMatrix b))
  (mv [_ alpha a x beta y]
    (sy-mv CBLAS/ssymv alpha ^RealUploMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sy-mv a))
  (rk [_ alpha x y a]
    (sy-r CBLAS/ssyr2 alpha ^RealBlockVector x ^RealBlockVector y ^RealUploMatrix a))
  (rk [_ alpha x a]
    (sy-r CBLAS/ssyr alpha ^RealBlockVector x ^RealUploMatrix a))
  (srk [_ alpha a beta c]
    (sy-rk CBLAS/ssyrk alpha ^RealGEMatrix a beta ^RealUploMatrix c))
  (mm [_ alpha a b beta c left]
    (sy-mm CBLAS/ssymm alpha ^RealUploMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ alpha a b _]
    (sy-mm a))
  BlasPlus
  (amax [_ a]
    (sy-lan LAPACK/slansy (int \M) ^RealUploMatrix a))
  (sum [_ a]
    (sy-sum CBLAS/ssum ^RealUploMatrix a))
  (set-all [_ alpha a]
    (sy-laset LAPACK/slaset alpha alpha ^RealUploMatrix a))
  (axpby [_ alpha a beta b]
    (sy-axpby MKL/saxpby alpha ^RealUploMatrix a beta ^RealUploMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for SY matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealUploMatrix a increasing))
  (trf [_ a ipiv]
    (sy-trx LAPACK/ssytrf ^RealUploMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (sy-trx LAPACK/spotrf ^RealUploMatrix a))
  (trfx [_ a]
    (sy-trfx LAPACK/spotrf ^RealUploMatrix a))
  (tri [_ ldl ipiv]
    (sy-trx LAPACK/ssytri ^RealUploMatrix ldl ^IntegerBlockVector ipiv))
  (tri [_ gg]
    (sy-trx LAPACK/spotri ^RealUploMatrix gg))
  (trs [_ ldl b ipiv]
    (sy-trs LAPACK/ssytrs ^RealUploMatrix ldl ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (trs [_ gg b]
    (sy-trs LAPACK/spotrs ^RealUploMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sy-sv LAPACK/sposv LAPACK/ssysv ^RealUploMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sy-sv LAPACK/sposv ^RealUploMatrix a ^RealGEMatrix b))
  (con [_ ldl ipiv nrm _]
    (sy-con LAPACK/ssycon ^RealUploMatrix ldl ^IntegerBlockVector ipiv nrm))
  (con [_ gg nrm _]
    (sy-con LAPACK/spocon ^RealUploMatrix gg nrm))
  (ev [_ a w vl vr]
    (let [v (or vl vr zero-matrix)]
      (sy-ev LAPACK/ssyevd LAPACK/ssyevr ^RealUploMatrix a ^RealGEMatrix w ^RealGEMatrix v)))
  VectorMath
  (sqr [_ a y]
    (matrix-math MKL/vsSqr ^RealUploMatrix a ^RealUploMatrix y))
  (mul [_ a b y]
    (matrix-math MKL/vsMul ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (div [_ a b y]
    (matrix-math MKL/vsDiv ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (inv [_ a y]
    (matrix-math MKL/vsInv ^RealUploMatrix a ^RealUploMatrix y))
  (abs [_ a y]
    (matrix-math MKL/vsAbs ^RealUploMatrix a ^RealUploMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (matrix-linear-frac MKL/vsLinearFrac ^RealUploMatrix a ^RealUploMatrix b
                        scalea shifta scaleb shiftb ^RealUploMatrix y))
  (fmod [_ a b y]
    (matrix-math MKL/vsFmod ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (frem [_ a b y]
    (matrix-math MKL/vsRemainder ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sqrt [_ a y]
    (matrix-math MKL/vsSqrt ^RealUploMatrix a ^RealUploMatrix y))
  (inv-sqrt [_ a y]
    (matrix-math MKL/vsInvSqrt ^RealUploMatrix a ^RealUploMatrix y))
  (cbrt [_ a y]
    (matrix-math MKL/vsCbrt ^RealUploMatrix a ^RealUploMatrix y))
  (inv-cbrt [_ a y]
    (matrix-math MKL/vsInvCbrt ^RealUploMatrix a ^RealUploMatrix y))
  (pow2o3 [_ a y]
    (matrix-math MKL/vsPow2o3 ^RealUploMatrix a ^RealUploMatrix y))
  (pow3o2 [_ a y]
    (matrix-math MKL/vsPow3o2 ^RealUploMatrix a ^RealUploMatrix y))
  (pow [_ a b y]
    (matrix-math MKL/vsPow ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (powx [_ a b y]
    (matrix-powx MKL/vsPowx ^RealUploMatrix a b ^RealUploMatrix y))
  (hypot [_ a b y]
    (matrix-math MKL/vsHypot ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (exp [_ a y]
    (matrix-math MKL/vsExp ^RealUploMatrix a ^RealUploMatrix y))
  (expm1 [_ a y]
    (matrix-math MKL/vsExpm1 ^RealUploMatrix a ^RealUploMatrix y))
  (log [_ a y]
    (matrix-math MKL/vsLn ^RealUploMatrix a ^RealUploMatrix y))
  (log10 [_ a y]
    (matrix-math MKL/vsLog10 ^RealUploMatrix a ^RealUploMatrix y))
  (sin [_ a y]
    (matrix-math MKL/vsSin ^RealUploMatrix a ^RealUploMatrix y))
  (cos [_ a y]
    (matrix-math MKL/vsCos ^RealUploMatrix a ^RealUploMatrix y))
  (tan [_ a y]
    (matrix-math MKL/vsTan ^RealUploMatrix a ^RealUploMatrix y))
  (sincos [_ a y z]
    (matrix-math MKL/vsSinCos ^RealUploMatrix a ^RealUploMatrix y ^RealUploMatrix z))
  (asin [_ a y]
    (matrix-math MKL/vsAsin ^RealUploMatrix a ^RealUploMatrix y))
  (acos [_ a y]
    (matrix-math MKL/vsAcos ^RealUploMatrix a ^RealUploMatrix y))
  (atan [_ a y]
    (matrix-math MKL/vsAtan ^RealUploMatrix a ^RealUploMatrix y))
  (atan2 [_ a b y]
    (matrix-math MKL/vsAtan2 ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sinh [_ a y]
    (matrix-math MKL/vsSinh ^RealUploMatrix a ^RealUploMatrix y))
  (cosh [_ a y]
    (matrix-math MKL/vsCosh ^RealUploMatrix a ^RealUploMatrix y))
  (tanh [_ a y]
    (matrix-math MKL/vsTanh ^RealUploMatrix a ^RealUploMatrix y))
  (asinh [_ a y]
    (matrix-math MKL/vsAsinh ^RealUploMatrix a ^RealUploMatrix y))
  (acosh [_ a y]
    (matrix-math MKL/vsAcosh ^RealUploMatrix a ^RealUploMatrix y))
  (atanh [_ a y]
    (matrix-math MKL/vsAtanh ^RealUploMatrix a ^RealUploMatrix y))
  (erf [_ a y]
    (matrix-math MKL/vsErf ^RealUploMatrix a ^RealUploMatrix y))
  (erfc [_ a y]
    (matrix-math MKL/vsErfc ^RealUploMatrix a ^RealUploMatrix y))
  (erf-inv [_ a y]
    (matrix-math MKL/vsErfInv ^RealUploMatrix a ^RealUploMatrix y))
  (erfc-inv [_ a y]
    (matrix-math MKL/vsErfcInv ^RealUploMatrix a ^RealUploMatrix y))
  (cdf-norm [_ a y]
    (matrix-math MKL/vsCdfNorm ^RealUploMatrix a ^RealUploMatrix y))
  (cdf-norm-inv [_ a y]
    (matrix-math MKL/vsCdfNormInv ^RealUploMatrix a ^RealUploMatrix y))
  (gamma [_ a y]
    (matrix-math MKL/vsGamma ^RealUploMatrix a ^RealUploMatrix y))
  (lgamma [_ a y]
    (matrix-math MKL/vsLGamma ^RealUploMatrix a ^RealUploMatrix y))
  (expint1 [_ a y]
    (matrix-math MKL/vsExpInt1 ^RealUploMatrix a ^RealUploMatrix y))
  (floor [_ a y]
    (matrix-math MKL/vsFloor ^RealUploMatrix a ^RealUploMatrix y))
  (fceil [_ a y]
    (matrix-math MKL/vsCeil ^RealUploMatrix a ^RealUploMatrix y))
  (trunc [_ a y]
    (matrix-math MKL/vsTrunc ^RealUploMatrix a ^RealUploMatrix y))
  (round [_ a y]
    (matrix-math MKL/vsRound ^RealUploMatrix a ^RealUploMatrix y))
  (modf [_ a y z]
    (matrix-math MKL/vsModf ^RealUploMatrix a ^RealUploMatrix y ^RealUploMatrix z))
  (frac [_ a y]
    (matrix-math MKL/vsFrac ^RealUploMatrix a ^RealUploMatrix y))
  (fmin [_ a b y]
    (matrix-math MKL/vsFmin ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (fmax [_ a b y]
    (matrix-math MKL/vsFmax ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (copy-sign [_ a b y]
    (matrix-math MKL/vsCopySign ^RealUploMatrix a ^RealUploMatrix b ^RealUploMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

;; =============== Banded Matrix Engines ===================================

(deftype DoubleGBEngine []
  Blas
  (swap [_ a b]
    (gb-map CBLAS/dswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (gb-map CBLAS/dcopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/dscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (gb-dot CBLAS/ddot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (gb-lan LAPACK/dlangb CBLAS/dnrm2 (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (gb-sum CBLAS/dasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (gb-axpy CBLAS/daxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ alpha a x beta y]
    (gb-mv CBLAS/dgbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (gb-mv a))
  (mm [_ alpha a b beta c left]
    (gb-mm CBLAS/dgbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (gb-mm a))
  BlasPlus
  (amax [_ a]
    (gb-lan LAPACK/dlangb CBLAS/idamax (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (gb-sum CBLAS/dsum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/dlaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (gb-axpby MKL/daxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealBandedMatrix a increasing))
  (trf [_ a ipiv]
    (gb-trf LAPACK/dgbtrf ^RealBandedMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (dragan-says-ex "Pivotless factorization is not available for GB matrices."))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ lu b ipiv]
    (gb-trs LAPACK/dgbtrs ^RealBandedMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (gb-sv LAPACK/dgbsv ^RealBandedMatrix a ^RealGEMatrix b pure))
  (con [_ ldl ipiv nrm nrm1?]
    (gb-con LAPACK/dgbcon ^RealBandedMatrix ldl ^IntegerBlockVector ipiv nrm nrm1?)))

(deftype FloatGBEngine []
  Blas
  (swap [_ a b]
    (gb-map CBLAS/sswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (gb-map CBLAS/scopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/sscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (gb-dot CBLAS/sdot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (gb-lan LAPACK/slangb CBLAS/idamax (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (gb-lan LAPACK/slangb CBLAS/snrm2 (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (gb-lan LAPACK/slangb CBLAS/idamax (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (gb-sum CBLAS/sasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (gb-axpy CBLAS/saxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ alpha a x beta y]
    (gb-mv CBLAS/sgbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (gb-mv a))
  (mm [_ alpha a b beta c left]
    (gb-mm CBLAS/sgbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (gb-mm a))
  BlasPlus
  (amax [_ a]
    (gb-lan LAPACK/slangb CBLAS/idamax (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (gb-sum CBLAS/ssum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/slaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (gb-axpby MKL/saxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealBandedMatrix a increasing))
  (trf [_ a ipiv]
    (gb-trf LAPACK/sgbtrf ^RealBandedMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (dragan-says-ex "Pivotless factorization is not available for GB matrices."))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ lu b ipiv]
    (gb-trs LAPACK/sgbtrs ^RealBandedMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (gb-sv LAPACK/sgbsv ^RealBandedMatrix a ^RealGEMatrix b pure))
  (con [_ ldl ipiv nrm nrm1?]
    (gb-con LAPACK/sgbcon ^RealBandedMatrix ldl ^IntegerBlockVector ipiv nrm nrm1?)))

(deftype DoubleSBEngine []
  Blas
  (swap [_ a b]
    (sb-map CBLAS/dswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (sb-map CBLAS/dcopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/dscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (sb-dot CBLAS/ddot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (sb-lan LAPACK/dlansb (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (sb-lan LAPACK/dlansb (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (sb-lan LAPACK/dlansb (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (sb-sum CBLAS/dasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (sb-axpy CBLAS/daxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ alpha a x beta y]
    (sb-mv CBLAS/dsbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (sb-mv a))
  (mm [_ alpha a b beta c left]
    (sb-mm CBLAS/dsbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sb-mm a))
  BlasPlus
  (amax [_ a]
    (sb-lan LAPACK/dlansb (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (sb-sum CBLAS/dsum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/dlaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (sb-axpby MKL/daxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealBandedMatrix a increasing))
  (trf [_ a]
    (sb-trf LAPACK/dpbtrf ^RealBandedMatrix a))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ gg b]
    (sb-trs LAPACK/dpbtrs ^RealBandedMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sb-sv LAPACK/dpbsv ^RealBandedMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sb-sv LAPACK/dpbsv ^RealBandedMatrix a ^RealGEMatrix b false))
  (con [_ gg nrm _]
    (sb-con LAPACK/dpbcon ^RealBandedMatrix gg nrm)))

(deftype FloatSBEngine []
  Blas
  (swap [_ a b]
    (sb-map CBLAS/sswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (sb-map CBLAS/scopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/sscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (sb-dot CBLAS/sdot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (sb-lan LAPACK/slansb (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (sb-lan LAPACK/slansb (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (sb-lan LAPACK/slansb (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (sb-sum CBLAS/sasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (sb-axpy CBLAS/saxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ alpha a x beta y]
    (sb-mv CBLAS/ssbmv alpha ^RealBandedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (sb-mv a))
  (mm [_ alpha a b beta c left]
    (sb-mm CBLAS/ssbmv alpha ^RealBandedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sb-mm a))
  BlasPlus
  (amax [_ a]
    (sb-lan LAPACK/slansb (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (sb-sum CBLAS/ssum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/slaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (sb-axpby MKL/saxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealBandedMatrix a increasing))
  (trf [_ a]
    (sb-trf LAPACK/spbtrf ^RealBandedMatrix a))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ gg b]
    (sb-trs LAPACK/spbtrs ^RealBandedMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sb-sv LAPACK/spbsv ^RealBandedMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sb-sv LAPACK/spbsv ^RealBandedMatrix a ^RealGEMatrix b false))
  (con [_ gg nrm _]
    (sb-con LAPACK/spbcon ^RealBandedMatrix gg nrm)))

(deftype DoubleTBEngine []
  Blas
  (swap [_ a b]
    (gb-map CBLAS/dswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (gb-map CBLAS/dcopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/dscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (tb-dot CBLAS/ddot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (tb-lan LAPACK/dlantb (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (tb-lan LAPACK/dlantb (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (tb-lan LAPACK/dlantb (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (tb-sum CBLAS/dasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (gb-axpy CBLAS/daxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ _ a _ _ _]
    (tb-mv a))
  (mv [_ a x]
    (tb-mv CBLAS/dtbmv ^RealBandedMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tb-mm a))
  (mm [_ alpha a b left]
    (tb-mm CBLAS/dtbmv alpha ^RealBandedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tb-lan LAPACK/dlantb (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (tb-sum CBLAS/dsum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/dlaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (gb-axpby MKL/daxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/dlasrt ^RealBandedMatrix a increasing))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ a b]
    (tb-trs LAPACK/dtbtrs ^RealBandedMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tb-sv CBLAS/dtbsv ^RealBandedMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tb-con LAPACK/dtbcon ^RealBandedMatrix a nrm1?)))

(deftype FloatTBEngine []
  Blas
  (swap [_ a b]
    (gb-map CBLAS/sswap ^RealBandedMatrix a ^RealBandedMatrix b)
    a)
  (copy [_ a b]
    (gb-map CBLAS/scopy ^RealBandedMatrix a ^RealBandedMatrix b))
  (scal [_ alpha a]
    (gb-scal CBLAS/sscal alpha ^RealBandedMatrix a))
  (dot [_ a b]
    (tb-dot CBLAS/sdot ^RealBandedMatrix a ^RealBandedMatrix b))
  (nrm1 [_ a]
    (tb-lan LAPACK/slantb (int \O) ^RealBandedMatrix a))
  (nrm2 [_ a]
    (tb-lan LAPACK/slantb (int \F) ^RealBandedMatrix a))
  (nrmi [_ a]
    (tb-lan LAPACK/slantb (int \I) ^RealBandedMatrix a))
  (asum [_ a]
    (tb-sum CBLAS/sasum ^RealBandedMatrix a))
  (axpy [_ alpha a b]
    (gb-axpy CBLAS/saxpy alpha ^RealBandedMatrix a ^RealBandedMatrix b))
  (mv [_ _ a _ _ _]
    (tb-mv a))
  (mv [_ a x]
    (tb-mv CBLAS/stbmv ^RealBandedMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tb-mm a))
  (mm [_ alpha a b left]
    (tb-mm CBLAS/stbmv alpha ^RealBandedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tb-lan LAPACK/slantb (int \M) ^RealBandedMatrix a))
  (sum [_ a]
    (tb-sum CBLAS/ssum ^RealBandedMatrix a))
  (set-all [_ alpha a]
    (gb-laset LAPACK/slaset alpha ^RealBandedMatrix a))
  (axpby [_ alpha a beta b]
    (gb-axpby MKL/saxpby alpha ^RealBandedMatrix a beta ^RealBandedMatrix b))
  (trans [_ a]
    (dragan-says-ex "In-place transpose is not available for banded matrices."))
  Lapack
  (srt [_ a increasing]
    (matrix-lasrt LAPACK/slasrt ^RealBandedMatrix a increasing))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for banded matrices."))
  (trs [_ a b]
    (tb-trs LAPACK/stbtrs ^RealBandedMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tb-sv CBLAS/stbsv ^RealBandedMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tb-con LAPACK/stbcon ^RealBandedMatrix a nrm1?)))

;; =============== Packed Matrix Engines ===================================

(deftype DoubleTPEngine []
  Blas
  (swap [_ a b]
    (packed-map CBLAS/dswap ^RealPackedMatrix a ^RealPackedMatrix b)
    a)
  (copy [_ a b]
    (packed-map CBLAS/dcopy ^RealPackedMatrix a ^RealPackedMatrix b))
  (scal [_ alpha a]
    (packed-scal CBLAS/dscal alpha ^RealPackedMatrix a))
  (dot [_ a b]
    (tp-dot CBLAS/ddot ^RealPackedMatrix a ^RealPackedMatrix b))
  (nrm1 [_ a]
    (tp-lan LAPACK/dlantp (int \O) ^RealPackedMatrix a))
  (nrm2 [_ a]
    (tp-lan LAPACK/dlantp (int \F) ^RealPackedMatrix a))
  (nrmi [_ a]
    (tp-lan LAPACK/dlantp (int \I) ^RealPackedMatrix a))
  (asum [_ a]
    (tp-sum CBLAS/dasum Math/abs ^RealPackedMatrix a))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/daxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [_ _ a _ _ _]
    (tp-mv a))
  (mv [_ a x]
    (tp-mv CBLAS/dtpmv ^RealPackedMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tp-mm a))
  (mm [_ alpha a b left]
    (tp-mm CBLAS/dtpmv alpha ^RealPackedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tp-lan LAPACK/dlantp (int \M) ^RealPackedMatrix a))
  (sum [_ a]
    (tp-sum CBLAS/ddot double ^RealPackedMatrix a ^RealBlockVector ones-double))
  (set-all [_ alpha a]
    (packed-laset LAPACK/dlaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/daxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/dlasrt ^RealPackedMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TP matrices."))
  (tri [_ a]
    (tp-tri LAPACK/dtptri ^RealPackedMatrix a))
  (trs [_ a b]
    (tp-trs LAPACK/dtptrs ^RealPackedMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tp-sv CBLAS/dtpsv ^RealPackedMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tp-con LAPACK/dtpcon ^RealPackedMatrix a nrm1?))
  VectorMath
  (sqr [_ a y]
    (packed-math MKL/vdSqr ^RealPackedMatrix a ^RealPackedMatrix y))
  (mul [_ a b y]
    (packed-math MKL/vdMul ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (div [_ a b y]
    (packed-math MKL/vdDiv ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (inv [_ a y]
    (packed-math MKL/vdInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (abs [_ a y]
    (packed-math MKL/vdAbs ^RealPackedMatrix a ^RealPackedMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (packed-linear-frac MKL/vdLinearFrac ^RealPackedMatrix a ^RealPackedMatrix b
                        scalea shifta scaleb shiftb ^RealPackedMatrix y))
  (fmod [_ a b y]
    (packed-math MKL/vdFmod ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (frem [_ a b y]
    (packed-math MKL/vdRemainder ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sqrt [_ a y]
    (packed-math MKL/vdSqrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (inv-sqrt [_ a y]
    (packed-math MKL/vdInvSqrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (cbrt [_ a y]
    (packed-math MKL/vdCbrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (inv-cbrt [_ a y]
    (packed-math MKL/vdInvCbrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow2o3 [_ a y]
    (packed-math MKL/vdPow2o3 ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow3o2 [_ a y]
    (packed-math MKL/vdPow3o2 ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow [_ a b y]
    (packed-math MKL/vdPow ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (powx [_ a b y]
    (packed-powx MKL/vdPowx ^RealPackedMatrix a b ^RealPackedMatrix y))
  (hypot [_ a b y]
    (packed-math MKL/vdHypot ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (exp [_ a y]
    (packed-math MKL/vdExp ^RealPackedMatrix a ^RealPackedMatrix y))
  (expm1 [_ a y]
    (packed-math MKL/vdExpm1 ^RealPackedMatrix a ^RealPackedMatrix y))
  (log [_ a y]
    (packed-math MKL/vdLn ^RealPackedMatrix a ^RealPackedMatrix y))
  (log10 [_ a y]
    (packed-math MKL/vdLog10 ^RealPackedMatrix a ^RealPackedMatrix y))
  (sin [_ a y]
    (packed-math MKL/vdSin ^RealPackedMatrix a ^RealPackedMatrix y))
  (cos [_ a y]
    (packed-math MKL/vdCos ^RealPackedMatrix a ^RealPackedMatrix y))
  (tan [_ a y]
    (packed-math MKL/vdTan ^RealPackedMatrix a ^RealPackedMatrix y))
  (sincos [_ a y z]
    (packed-math MKL/vdSinCos ^RealPackedMatrix a ^RealPackedMatrix y ^RealPackedMatrix z))
  (asin [_ a y]
    (packed-math MKL/vdAsin ^RealPackedMatrix a ^RealPackedMatrix y))
  (acos [_ a y]
    (packed-math MKL/vdAcos ^RealPackedMatrix a ^RealPackedMatrix y))
  (atan [_ a y]
    (packed-math MKL/vdAtan ^RealPackedMatrix a ^RealPackedMatrix y))
  (atan2 [_ a b y]
    (packed-math MKL/vdAtan2 ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sinh [_ a y]
    (packed-math MKL/vdSinh ^RealPackedMatrix a ^RealPackedMatrix y))
  (cosh [_ a y]
    (packed-math MKL/vdCosh ^RealPackedMatrix a ^RealPackedMatrix y))
  (tanh [_ a y]
    (packed-math MKL/vdTanh ^RealPackedMatrix a ^RealPackedMatrix y))
  (asinh [_ a y]
    (packed-math MKL/vdAsinh ^RealPackedMatrix a ^RealPackedMatrix y))
  (acosh [_ a y]
    (packed-math MKL/vdAcosh ^RealPackedMatrix a ^RealPackedMatrix y))
  (atanh [_ a y]
    (packed-math MKL/vdAtanh ^RealPackedMatrix a ^RealPackedMatrix y))
  (erf [_ a y]
    (packed-math MKL/vdErf ^RealPackedMatrix a ^RealPackedMatrix y))
  (erfc [_ a y]
    (packed-math MKL/vdErfc ^RealPackedMatrix a ^RealPackedMatrix y))
  (erf-inv [_ a y]
    (packed-math MKL/vdErfInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (erfc-inv [_ a y]
    (packed-math MKL/vdErfcInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (cdf-norm [_ a y]
    (packed-math MKL/vdCdfNorm ^RealPackedMatrix a ^RealPackedMatrix y))
  (cdf-norm-inv [_ a y]
    (packed-math MKL/vdCdfNormInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (gamma [_ a y]
    (packed-math MKL/vdGamma ^RealPackedMatrix a ^RealPackedMatrix y))
  (lgamma [_ a y]
    (packed-math MKL/vdLGamma ^RealPackedMatrix a ^RealPackedMatrix y))
  (expint1 [_ a y]
    (packed-math MKL/vdExpInt1 ^RealPackedMatrix a ^RealPackedMatrix y))
  (floor [_ a y]
    (packed-math MKL/vdFloor ^RealPackedMatrix a ^RealPackedMatrix y))
  (fceil [_ a y]
    (packed-math MKL/vdCeil ^RealPackedMatrix a ^RealPackedMatrix y))
  (trunc [_ a y]
    (packed-math MKL/vdTrunc ^RealPackedMatrix a ^RealPackedMatrix y))
  (round [_ a y]
    (packed-math MKL/vdRound ^RealPackedMatrix a ^RealPackedMatrix y))
  (modf [_ a y z]
    (packed-math MKL/vdModf ^RealPackedMatrix a ^RealPackedMatrix y ^RealPackedMatrix z))
  (frac [_ a y]
    (packed-math MKL/vdFrac ^RealPackedMatrix a ^RealPackedMatrix y))
  (fmin [_ a b y]
    (packed-math MKL/vdFmin ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (fmax [_ a b y]
    (packed-math MKL/vdFmax ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (copy-sign [_ a b y]
    (packed-math MKL/vdCopySign ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype FloatTPEngine []
  Blas
  (swap [_ a b]
    (packed-map CBLAS/sswap ^RealPackedMatrix a ^RealPackedMatrix b)
    a)
  (copy [_ a b]
    (packed-map CBLAS/scopy ^RealPackedMatrix a ^RealPackedMatrix b))
  (scal [_ alpha a]
    (packed-scal CBLAS/sscal alpha ^RealPackedMatrix a))
  (dot [_ a b]
    (tp-dot CBLAS/sdot ^RealPackedMatrix a ^RealPackedMatrix b))
  (nrm1 [_ a]
    (tp-lan LAPACK/slantp (int \O) ^RealPackedMatrix a))
  (nrm2 [_ a]
    (tp-lan LAPACK/slantp (int \F) ^RealPackedMatrix a))
  (nrmi [_ a]
    (tp-lan LAPACK/slantp (int \I) ^RealPackedMatrix a))
  (asum [_ a]
    (tp-sum CBLAS/sasum Math/abs ^RealPackedMatrix a))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/saxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [_ _ a _ _ _]
    (tp-mv a))
  (mv [_ a x]
    (tp-mv CBLAS/stpmv ^RealPackedMatrix a ^RealBlockVector x))
  (mm [_ _ a _ _ _ _]
    (tp-mm a))
  (mm [_ alpha a b left]
    (tp-mm CBLAS/stpmv alpha ^RealPackedMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (tp-lan LAPACK/slantp (int \M) ^RealPackedMatrix a))
  (sum [_ a]
    (tp-sum CBLAS/sdot double ^RealPackedMatrix a ^RealBlockVector ones-float))
  (set-all [_ alpha a]
    (packed-laset LAPACK/slaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/saxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/slasrt ^RealPackedMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "There is no use for pivots when working with TP matrices."))
  (tri [_ a]
    (tp-tri LAPACK/stptri ^RealPackedMatrix a))
  (trs [_ a b]
    (tp-trs LAPACK/stptrs ^RealPackedMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (tp-sv CBLAS/stpsv ^RealPackedMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (tp-con LAPACK/stpcon ^RealPackedMatrix a nrm1?))
  VectorMath
  (sqr [_ a y]
    (packed-math MKL/vsSqr ^RealPackedMatrix a ^RealPackedMatrix y))
  (mul [_ a b y]
    (packed-math MKL/vsMul ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (div [_ a b y]
    (packed-math MKL/vsDiv ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (inv [_ a y]
    (packed-math MKL/vsInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (abs [_ a y]
    (packed-math MKL/vsAbs ^RealPackedMatrix a ^RealPackedMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (packed-linear-frac MKL/vsLinearFrac ^RealPackedMatrix a ^RealPackedMatrix b
                        scalea shifta scaleb shiftb ^RealPackedMatrix y))
  (fmod [_ a b y]
    (packed-math MKL/vsFmod ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (frem [_ a b y]
    (packed-math MKL/vsRemainder ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sqrt [_ a y]
    (packed-math MKL/vsSqrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (inv-sqrt [_ a y]
    (packed-math MKL/vsInvSqrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (cbrt [_ a y]
    (packed-math MKL/vsCbrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (inv-cbrt [_ a y]
    (packed-math MKL/vsInvCbrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow2o3 [_ a y]
    (packed-math MKL/vsPow2o3 ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow3o2 [_ a y]
    (packed-math MKL/vsPow3o2 ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow [_ a b y]
    (packed-math MKL/vsPow ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (powx [_ a b y]
    (packed-powx MKL/vsPowx ^RealPackedMatrix a b ^RealPackedMatrix y))
  (hypot [_ a b y]
    (packed-math MKL/vsHypot ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (exp [_ a y]
    (packed-math MKL/vsExp ^RealPackedMatrix a ^RealPackedMatrix y))
  (expm1 [_ a y]
    (packed-math MKL/vsExpm1 ^RealPackedMatrix a ^RealPackedMatrix y))
  (log [_ a y]
    (packed-math MKL/vsLn ^RealPackedMatrix a ^RealPackedMatrix y))
  (log10 [_ a y]
    (packed-math MKL/vsLog10 ^RealPackedMatrix a ^RealPackedMatrix y))
  (sin [_ a y]
    (packed-math MKL/vsSin ^RealPackedMatrix a ^RealPackedMatrix y))
  (cos [_ a y]
    (packed-math MKL/vsCos ^RealPackedMatrix a ^RealPackedMatrix y))
  (tan [_ a y]
    (packed-math MKL/vsTan ^RealPackedMatrix a ^RealPackedMatrix y))
  (sincos [_ a y z]
    (packed-math MKL/vsSinCos ^RealPackedMatrix a ^RealPackedMatrix y ^RealPackedMatrix z))
  (asin [_ a y]
    (packed-math MKL/vsAsin ^RealPackedMatrix a ^RealPackedMatrix y))
  (acos [_ a y]
    (packed-math MKL/vsAcos ^RealPackedMatrix a ^RealPackedMatrix y))
  (atan [_ a y]
    (packed-math MKL/vsAtan ^RealPackedMatrix a ^RealPackedMatrix y))
  (atan2 [_ a b y]
    (packed-math MKL/vsAtan2 ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sinh [_ a y]
    (packed-math MKL/vsSinh ^RealPackedMatrix a ^RealPackedMatrix y))
  (cosh [_ a y]
    (packed-math MKL/vsCosh ^RealPackedMatrix a ^RealPackedMatrix y))
  (tanh [_ a y]
    (packed-math MKL/vsTanh ^RealPackedMatrix a ^RealPackedMatrix y))
  (asinh [_ a y]
    (packed-math MKL/vsAsinh ^RealPackedMatrix a ^RealPackedMatrix y))
  (acosh [_ a y]
    (packed-math MKL/vsAcosh ^RealPackedMatrix a ^RealPackedMatrix y))
  (atanh [_ a y]
    (packed-math MKL/vsAtanh ^RealPackedMatrix a ^RealPackedMatrix y))
  (erf [_ a y]
    (packed-math MKL/vsErf ^RealPackedMatrix a ^RealPackedMatrix y))
  (erfc [_ a y]
    (packed-math MKL/vsErfc ^RealPackedMatrix a ^RealPackedMatrix y))
  (erf-inv [_ a y]
    (packed-math MKL/vsErfInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (erfc-inv [_ a y]
    (packed-math MKL/vsErfcInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (cdf-norm [_ a y]
    (packed-math MKL/vsCdfNorm ^RealPackedMatrix a ^RealPackedMatrix y))
  (cdf-norm-inv [_ a y]
    (packed-math MKL/vsCdfNormInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (gamma [_ a y]
    (packed-math MKL/vsGamma ^RealPackedMatrix a ^RealPackedMatrix y))
  (lgamma [_ a y]
    (packed-math MKL/vsLGamma ^RealPackedMatrix a ^RealPackedMatrix y))
  (expint1 [_ a y]
    (packed-math MKL/vsExpInt1 ^RealPackedMatrix a ^RealPackedMatrix y))
  (floor [_ a y]
    (packed-math MKL/vsFloor ^RealPackedMatrix a ^RealPackedMatrix y))
  (fceil [_ a y]
    (packed-math MKL/vsCeil ^RealPackedMatrix a ^RealPackedMatrix y))
  (trunc [_ a y]
    (packed-math MKL/vsTrunc ^RealPackedMatrix a ^RealPackedMatrix y))
  (round [_ a y]
    (packed-math MKL/vsRound ^RealPackedMatrix a ^RealPackedMatrix y))
  (modf [_ a y z]
    (packed-math MKL/vsModf ^RealPackedMatrix a ^RealPackedMatrix y ^RealPackedMatrix z))
  (frac [_ a y]
    (packed-math MKL/vsFrac ^RealPackedMatrix a ^RealPackedMatrix y))
  (fmin [_ a b y]
    (packed-math MKL/vsFmin ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (fmax [_ a b y]
    (packed-math MKL/vsFmax ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (copy-sign [_ a b y]
    (packed-math MKL/vsCopySign ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype DoubleSPEngine []
  Blas
  (swap [_ a b]
    (packed-map CBLAS/dswap ^RealPackedMatrix a ^RealPackedMatrix b)
    a)
  (copy [_ a b]
    (packed-map CBLAS/dcopy ^RealPackedMatrix a ^RealPackedMatrix b))
  (scal [_ alpha a]
    (packed-scal CBLAS/dscal alpha ^RealPackedMatrix a))
  (dot [_ a b]
    (sp-dot CBLAS/ddot ^RealPackedMatrix a ^RealPackedMatrix b))
  (nrm1 [_ a]
    (sp-lan LAPACK/dlansp (int \O) ^RealPackedMatrix a))
  (nrm2 [_ a]
    (sp-lan LAPACK/dlansp (int \F) ^RealPackedMatrix a))
  (nrmi [_ a]
    (sp-lan LAPACK/dlansp (int \I) ^RealPackedMatrix a))
  (asum [_ a]
    (sp-sum CBLAS/dasum Math/abs ^RealPackedMatrix a))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/daxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [_ alpha a x beta y]
    (sp-mv CBLAS/dspmv alpha ^RealPackedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sp-mv a))
  (rk [_ alpha x y a]
    (sp-r CBLAS/dspr2 alpha ^RealBlockVector x ^RealBlockVector y ^RealPackedMatrix a))
  (rk [_ alpha x a]
    (sp-r CBLAS/dspr alpha ^RealBlockVector x ^RealPackedMatrix a))
  (mm [_ alpha a b beta c left]
    (sp-mm CBLAS/dspmv alpha ^RealPackedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sp-mm a))
  BlasPlus
  (amax [_ a]
    (sp-lan LAPACK/dlansp (int \M) ^RealPackedMatrix a))
  (sum [_ a]
    (sp-sum CBLAS/ddot double ^RealPackedMatrix a ^RealBlockVector ones-double))
  (set-all [_ alpha a]
    (packed-laset LAPACK/dlaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/daxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/dlasrt ^RealPackedMatrix a increasing))
  (trf [_ a ipiv]
    (sp-trx LAPACK/dsptrf ^RealPackedMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (sp-trx LAPACK/dpptrf ^RealPackedMatrix a))
  (trfx [_ a]
    (sp-trfx LAPACK/dpptrf ^RealPackedMatrix a))
  (tri [_ ldl ipiv]
    (sp-trx LAPACK/dsptri ^RealPackedMatrix ldl ^IntegerBlockVector ipiv))
  (tri [_ gg]
    (sp-trx LAPACK/dpptri ^RealPackedMatrix gg))
  (trs [_ ldl b ipiv]
    (sp-trs LAPACK/dsptrs ^RealPackedMatrix ldl ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (trs [_ gg b]
    (sp-trs LAPACK/dpptrs ^RealPackedMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sp-sv LAPACK/dppsv LAPACK/dspsv ^RealPackedMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sp-sv LAPACK/dppsv ^RealPackedMatrix a ^RealGEMatrix b))
  (con [_ ldl ipiv nrm _]
    (sp-con LAPACK/dspcon ^RealPackedMatrix ldl ^IntegerBlockVector ipiv nrm))
  (con [_ gg nrm _]
    (sp-con LAPACK/dppcon ^RealPackedMatrix gg nrm))
  VectorMath
  (sqr [_ a y]
    (packed-math MKL/vdSqr ^RealPackedMatrix a ^RealPackedMatrix y))
  (mul [_ a b y]
    (packed-math MKL/vdMul ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (div [_ a b y]
    (packed-math MKL/vdDiv ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (inv [_ a y]
    (packed-math MKL/vdInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (abs [_ a y]
    (packed-math MKL/vdAbs ^RealPackedMatrix a ^RealPackedMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (packed-linear-frac MKL/vdLinearFrac ^RealPackedMatrix a ^RealPackedMatrix b
                        scalea shifta scaleb shiftb ^RealPackedMatrix y))
  (fmod [_ a b y]
    (packed-math MKL/vdFmod ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (frem [_ a b y]
    (packed-math MKL/vdRemainder ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sqrt [_ a y]
    (packed-math MKL/vdSqrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (inv-sqrt [_ a y]
    (packed-math MKL/vdInvSqrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (cbrt [_ a y]
    (packed-math MKL/vdCbrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (inv-cbrt [_ a y]
    (packed-math MKL/vdInvCbrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow2o3 [_ a y]
    (packed-math MKL/vdPow2o3 ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow3o2 [_ a y]
    (packed-math MKL/vdPow3o2 ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow [_ a b y]
    (packed-math MKL/vdPow ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (powx [_ a b y]
    (packed-powx MKL/vdPowx ^RealPackedMatrix a b ^RealPackedMatrix y))
  (hypot [_ a b y]
    (packed-math MKL/vdHypot ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (exp [_ a y]
    (packed-math MKL/vdExp ^RealPackedMatrix a ^RealPackedMatrix y))
  (expm1 [_ a y]
    (packed-math MKL/vdExpm1 ^RealPackedMatrix a ^RealPackedMatrix y))
  (log [_ a y]
    (packed-math MKL/vdLn ^RealPackedMatrix a ^RealPackedMatrix y))
  (log10 [_ a y]
    (packed-math MKL/vdLog10 ^RealPackedMatrix a ^RealPackedMatrix y))
  (sin [_ a y]
    (packed-math MKL/vdSin ^RealPackedMatrix a ^RealPackedMatrix y))
  (cos [_ a y]
    (packed-math MKL/vdCos ^RealPackedMatrix a ^RealPackedMatrix y))
  (tan [_ a y]
    (packed-math MKL/vdTan ^RealPackedMatrix a ^RealPackedMatrix y))
  (sincos [_ a y z]
    (packed-math MKL/vdSinCos ^RealPackedMatrix a ^RealPackedMatrix y ^RealPackedMatrix z))
  (asin [_ a y]
    (packed-math MKL/vdAsin ^RealPackedMatrix a ^RealPackedMatrix y))
  (acos [_ a y]
    (packed-math MKL/vdAcos ^RealPackedMatrix a ^RealPackedMatrix y))
  (atan [_ a y]
    (packed-math MKL/vdAtan ^RealPackedMatrix a ^RealPackedMatrix y))
  (atan2 [_ a b y]
    (packed-math MKL/vdAtan2 ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sinh [_ a y]
    (packed-math MKL/vdSinh ^RealPackedMatrix a ^RealPackedMatrix y))
  (cosh [_ a y]
    (packed-math MKL/vdCosh ^RealPackedMatrix a ^RealPackedMatrix y))
  (tanh [_ a y]
    (packed-math MKL/vdTanh ^RealPackedMatrix a ^RealPackedMatrix y))
  (asinh [_ a y]
    (packed-math MKL/vdAsinh ^RealPackedMatrix a ^RealPackedMatrix y))
  (acosh [_ a y]
    (packed-math MKL/vdAcosh ^RealPackedMatrix a ^RealPackedMatrix y))
  (atanh [_ a y]
    (packed-math MKL/vdAtanh ^RealPackedMatrix a ^RealPackedMatrix y))
  (erf [_ a y]
    (packed-math MKL/vdErf ^RealPackedMatrix a ^RealPackedMatrix y))
  (erfc [_ a y]
    (packed-math MKL/vdErfc ^RealPackedMatrix a ^RealPackedMatrix y))
  (erf-inv [_ a y]
    (packed-math MKL/vdErfInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (erfc-inv [_ a y]
    (packed-math MKL/vdErfcInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (cdf-norm [_ a y]
    (packed-math MKL/vdCdfNorm ^RealPackedMatrix a ^RealPackedMatrix y))
  (cdf-norm-inv [_ a y]
    (packed-math MKL/vdCdfNormInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (gamma [_ a y]
    (packed-math MKL/vdGamma ^RealPackedMatrix a ^RealPackedMatrix y))
  (lgamma [_ a y]
    (packed-math MKL/vdLGamma ^RealPackedMatrix a ^RealPackedMatrix y))
  (expint1 [_ a y]
    (packed-math MKL/vdExpInt1 ^RealPackedMatrix a ^RealPackedMatrix y))
  (floor [_ a y]
    (packed-math MKL/vdFloor ^RealPackedMatrix a ^RealPackedMatrix y))
  (fceil [_ a y]
    (packed-math MKL/vdCeil ^RealPackedMatrix a ^RealPackedMatrix y))
  (trunc [_ a y]
    (packed-math MKL/vdTrunc ^RealPackedMatrix a ^RealPackedMatrix y))
  (round [_ a y]
    (packed-math MKL/vdRound ^RealPackedMatrix a ^RealPackedMatrix y))
  (modf [_ a y z]
    (packed-math MKL/vdModf ^RealPackedMatrix a ^RealPackedMatrix y ^RealPackedMatrix z))
  (frac [_ a y]
    (packed-math MKL/vdFrac ^RealPackedMatrix a ^RealPackedMatrix y))
  (fmin [_ a b y]
    (packed-math MKL/vdFmin ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (fmax [_ a b y]
    (packed-math MKL/vdFmax ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (copy-sign [_ a b y]
    (packed-math MKL/vdCopySign ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype FloatSPEngine []
  Blas
  (swap [_ a b]
    (packed-map CBLAS/sswap ^RealPackedMatrix a ^RealPackedMatrix b)
    a)
  (copy [_ a b]
    (packed-map CBLAS/scopy ^RealPackedMatrix a ^RealPackedMatrix b))
  (scal [_ alpha a]
    (packed-scal CBLAS/sscal alpha ^RealPackedMatrix a))
  (dot [_ a b]
    (sp-dot CBLAS/sdot ^RealPackedMatrix a ^RealPackedMatrix b))
  (nrm1 [_ a]
    (sp-lan LAPACK/slansp (int \O) ^RealPackedMatrix a))
  (nrm2 [_ a]
    (sp-lan LAPACK/slansp (int \F) ^RealPackedMatrix a))
  (nrmi [_ a]
    (sp-lan LAPACK/slansp (int \I) ^RealPackedMatrix a))
  (asum [_ a]
    (sp-sum CBLAS/sasum Math/abs ^RealPackedMatrix a))
  (axpy [_ alpha a b]
    (packed-axpy CBLAS/saxpy alpha ^RealPackedMatrix a ^RealPackedMatrix b))
  (mv [_ alpha a x beta y]
    (sp-mv CBLAS/sspmv alpha ^RealPackedMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (sp-mv a))
  (rk [_ alpha x y a]
    (sp-r CBLAS/sspr2 alpha ^RealBlockVector x ^RealBlockVector y ^RealPackedMatrix a))
  (rk [_ alpha x a]
    (sp-r CBLAS/sspr alpha ^RealBlockVector x ^RealPackedMatrix a))
  (mm [_ alpha a b beta c left]
    (sp-mm CBLAS/sspmv alpha ^RealPackedMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (sp-mm a))
  BlasPlus
  (amax [_ a]
    (sp-lan LAPACK/slansp (int \M) ^RealPackedMatrix a))
  (sum [_ a]
    (sp-sum CBLAS/sdot double ^RealPackedMatrix a ^RealBlockVector ones-float))
  (set-all [_ alpha a]
    (packed-laset LAPACK/slaset alpha ^RealPackedMatrix a))
  (axpby [_ alpha a beta b]
    (packed-axpby MKL/saxpby alpha ^RealPackedMatrix a beta ^RealPackedMatrix b))
  Lapack
  (srt [_ a increasing]
    (packed-lasrt LAPACK/slasrt ^RealPackedMatrix a increasing))
  (trf [_ a ipiv]
    (sp-trx LAPACK/ssptrf ^RealPackedMatrix a ^IntegerBlockVector ipiv))
  (trf [_ a]
    (sp-trx LAPACK/spptrf ^RealPackedMatrix a))
  (trfx [_ a]
    (sp-trfx LAPACK/spptrf ^RealPackedMatrix a))
  (tri [_ ldl ipiv]
    (sp-trx LAPACK/ssptri ^RealPackedMatrix ldl ^IntegerBlockVector ipiv))
  (tri [_ gg]
    (sp-trx LAPACK/spptri ^RealPackedMatrix gg))
  (trs [_ ldl b ipiv]
    (sp-trs LAPACK/ssptrs ^RealPackedMatrix ldl ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (trs [_ gg b]
    (sp-trs LAPACK/spptrs ^RealPackedMatrix gg ^RealGEMatrix b))
  (sv [_ a b pure]
    (sp-sv LAPACK/sppsv LAPACK/sspsv ^RealPackedMatrix a ^RealGEMatrix b pure))
  (sv [_ a b]
    (sp-sv LAPACK/sppsv ^RealPackedMatrix a ^RealGEMatrix b))
  (con [_ ldl ipiv nrm _]
    (sp-con LAPACK/sspcon ^RealPackedMatrix ldl ^IntegerBlockVector ipiv nrm))
  (con [_ gg nrm _]
    (sp-con LAPACK/sppcon ^RealPackedMatrix gg nrm))
  VectorMath
  (sqr [_ a y]
    (packed-math MKL/vsSqr ^RealPackedMatrix a ^RealPackedMatrix y))
  (mul [_ a b y]
    (packed-math MKL/vsMul ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (div [_ a b y]
    (packed-math MKL/vsDiv ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (inv [_ a y]
    (packed-math MKL/vsInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (abs [_ a y]
    (packed-math MKL/vsAbs ^RealPackedMatrix a ^RealPackedMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (packed-linear-frac MKL/vsLinearFrac ^RealPackedMatrix a ^RealPackedMatrix b
                        scalea shifta scaleb shiftb ^RealPackedMatrix y))
  (fmod [_ a b y]
    (packed-math MKL/vsFmod ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (frem [_ a b y]
    (packed-math MKL/vsRemainder ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sqrt [_ a y]
    (packed-math MKL/vsSqrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (inv-sqrt [_ a y]
    (packed-math MKL/vsInvSqrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (cbrt [_ a y]
    (packed-math MKL/vsCbrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (inv-cbrt [_ a y]
    (packed-math MKL/vsInvCbrt ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow2o3 [_ a y]
    (packed-math MKL/vsPow2o3 ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow3o2 [_ a y]
    (packed-math MKL/vsPow3o2 ^RealPackedMatrix a ^RealPackedMatrix y))
  (pow [_ a b y]
    (packed-math MKL/vsPow ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (powx [_ a b y]
    (packed-powx MKL/vsPowx ^RealPackedMatrix a b ^RealPackedMatrix y))
  (hypot [_ a b y]
    (packed-math MKL/vsHypot ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (exp [_ a y]
    (packed-math MKL/vsExp ^RealPackedMatrix a ^RealPackedMatrix y))
  (expm1 [_ a y]
    (packed-math MKL/vsExpm1 ^RealPackedMatrix a ^RealPackedMatrix y))
  (log [_ a y]
    (packed-math MKL/vsLn ^RealPackedMatrix a ^RealPackedMatrix y))
  (log10 [_ a y]
    (packed-math MKL/vsLog10 ^RealPackedMatrix a ^RealPackedMatrix y))
  (sin [_ a y]
    (packed-math MKL/vsSin ^RealPackedMatrix a ^RealPackedMatrix y))
  (cos [_ a y]
    (packed-math MKL/vsCos ^RealPackedMatrix a ^RealPackedMatrix y))
  (tan [_ a y]
    (packed-math MKL/vsTan ^RealPackedMatrix a ^RealPackedMatrix y))
  (sincos [_ a y z]
    (packed-math MKL/vsSinCos ^RealPackedMatrix a ^RealPackedMatrix y ^RealPackedMatrix z))
  (asin [_ a y]
    (packed-math MKL/vsAsin ^RealPackedMatrix a ^RealPackedMatrix y))
  (acos [_ a y]
    (packed-math MKL/vsAcos ^RealPackedMatrix a ^RealPackedMatrix y))
  (atan [_ a y]
    (packed-math MKL/vsAtan ^RealPackedMatrix a ^RealPackedMatrix y))
  (atan2 [_ a b y]
    (packed-math MKL/vsAtan2 ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sinh [_ a y]
    (packed-math MKL/vsSinh ^RealPackedMatrix a ^RealPackedMatrix y))
  (cosh [_ a y]
    (packed-math MKL/vsCosh ^RealPackedMatrix a ^RealPackedMatrix y))
  (tanh [_ a y]
    (packed-math MKL/vsTanh ^RealPackedMatrix a ^RealPackedMatrix y))
  (asinh [_ a y]
    (packed-math MKL/vsAsinh ^RealPackedMatrix a ^RealPackedMatrix y))
  (acosh [_ a y]
    (packed-math MKL/vsAcosh ^RealPackedMatrix a ^RealPackedMatrix y))
  (atanh [_ a y]
    (packed-math MKL/vsAtanh ^RealPackedMatrix a ^RealPackedMatrix y))
  (erf [_ a y]
    (packed-math MKL/vsErf ^RealPackedMatrix a ^RealPackedMatrix y))
  (erfc [_ a y]
    (packed-math MKL/vsErfc ^RealPackedMatrix a ^RealPackedMatrix y))
  (erf-inv [_ a y]
    (packed-math MKL/vsErfInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (erfc-inv [_ a y]
    (packed-math MKL/vsErfcInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (cdf-norm [_ a y]
    (packed-math MKL/vsCdfNorm ^RealPackedMatrix a ^RealPackedMatrix y))
  (cdf-norm-inv [_ a y]
    (packed-math MKL/vsCdfNormInv ^RealPackedMatrix a ^RealPackedMatrix y))
  (gamma [_ a y]
    (packed-math MKL/vsGamma ^RealPackedMatrix a ^RealPackedMatrix y))
  (lgamma [_ a y]
    (packed-math MKL/vsLGamma ^RealPackedMatrix a ^RealPackedMatrix y))
  (expint1 [_ a y]
    (packed-math MKL/vsExpInt1 ^RealPackedMatrix a ^RealPackedMatrix y))
  (floor [_ a y]
    (packed-math MKL/vsFloor ^RealPackedMatrix a ^RealPackedMatrix y))
  (fceil [_ a y]
    (packed-math MKL/vsCeil ^RealPackedMatrix a ^RealPackedMatrix y))
  (trunc [_ a y]
    (packed-math MKL/vsTrunc ^RealPackedMatrix a ^RealPackedMatrix y))
  (round [_ a y]
    (packed-math MKL/vsRound ^RealPackedMatrix a ^RealPackedMatrix y))
  (modf [_ a y z]
    (packed-math MKL/vsModf ^RealPackedMatrix a ^RealPackedMatrix y ^RealPackedMatrix z))
  (frac [_ a y]
    (packed-math MKL/vsFrac ^RealPackedMatrix a ^RealPackedMatrix y))
  (fmin [_ a b y]
    (packed-math MKL/vsFmin ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (fmax [_ a b y]
    (packed-math MKL/vsFmax ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (copy-sign [_ a b y]
    (packed-math MKL/vsCopySign ^RealPackedMatrix a ^RealPackedMatrix b ^RealPackedMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

;; =============== Tridiagonal Matrix Engines =================================

(deftype DoubleGTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/dswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/dcopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/dscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/ddot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/dnrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/dasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/daxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/dlagtm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/dlagtm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/dsum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/dlaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/daxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/dlasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a ipiv]
    (diagonal-trf LAPACK/dgttrf ^RealDiagonalMatrix a ^IntegerBlockVector ipiv))
  (trf [_ _]
    (dragan-says-ex "Pivotless factorization is not available for GT matrices."))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for GT matrices."))
  (trs [_ lu b ipiv]
    (gt-trs LAPACK/dgttrs ^RealDiagonalMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (gt-sv LAPACK/dgtsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (gt-con LAPACK/dgtcon ^RealDiagonalMatrix lu ^IntegerBlockVector ipiv nrm nrm1?))
  VectorMath
  (sqr [_ a y]
    (diagonal-math MKL/vdSqr ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (mul [_ a b y]
    (diagonal-math MKL/vdMul ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (div [_ a b y]
    (diagonal-math MKL/vdDiv ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (inv [_ a y]
    (diagonal-math MKL/vdInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (abs [_ a y]
    (diagonal-math MKL/vdAbs ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (diagonal-linear-frac MKL/vdLinearFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix b
                          scalea shifta scaleb shiftb ^RealDiagonalMatrix y))
  (fmod [_ a b y]
    (diagonal-math MKL/vdFmod ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (frem [_ a b y]
    (diagonal-math MKL/vdRemainder ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sqrt [_ a y]
    (diagonal-math MKL/vdSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-sqrt [_ a y]
    (diagonal-math MKL/vdInvSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cbrt [_ a y]
    (diagonal-math MKL/vdCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-cbrt [_ a y]
    (diagonal-math MKL/vdInvCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow2o3 [_ a y]
    (diagonal-math MKL/vdPow2o3 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow3o2 [_ a y]
    (diagonal-math MKL/vdPow3o2 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow [_ a b y]
    (diagonal-math MKL/vdPow ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (powx [_ a b y]
    (diagonal-powx MKL/vdPowx ^RealDiagonalMatrix a b ^RealDiagonalMatrix y))
  (hypot [_ a b y]
    (diagonal-math MKL/vdHypot ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (exp [_ a y]
    (diagonal-math MKL/vdExp ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expm1 [_ a y]
    (diagonal-math MKL/vdExpm1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log [_ a y]
    (diagonal-math MKL/vdLn ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log10 [_ a y]
    (diagonal-math MKL/vdLog10 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sin [_ a y]
    (diagonal-math MKL/vdSin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cos [_ a y]
    (diagonal-math MKL/vdCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tan [_ a y]
    (diagonal-math MKL/vdTan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sincos [_ a y z]
    (diagonal-math MKL/vdSinCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (asin [_ a y]
    (diagonal-math MKL/vdAsin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acos [_ a y]
    (diagonal-math MKL/vdAcos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan [_ a y]
    (diagonal-math MKL/vdAtan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan2 [_ a b y]
    (diagonal-math MKL/vdAtan2 ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sinh [_ a y]
    (diagonal-math MKL/vdSinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cosh [_ a y]
    (diagonal-math MKL/vdCosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tanh [_ a y]
    (diagonal-math MKL/vdTanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (asinh [_ a y]
    (diagonal-math MKL/vdAsinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acosh [_ a y]
    (diagonal-math MKL/vdAcosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atanh [_ a y]
    (diagonal-math MKL/vdAtanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf [_ a y]
    (diagonal-math MKL/vdErf ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc [_ a y]
    (diagonal-math MKL/vdErfc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf-inv [_ a y]
    (diagonal-math MKL/vdErfInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc-inv [_ a y]
    (diagonal-math MKL/vdErfcInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm [_ a y]
    (diagonal-math MKL/vdCdfNorm ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm-inv [_ a y]
    (diagonal-math MKL/vdCdfNormInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (gamma [_ a y]
    (diagonal-math MKL/vdGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (lgamma [_ a y]
    (diagonal-math MKL/vdLGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expint1 [_ a y]
    (diagonal-math MKL/vdExpInt1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (floor [_ a y]
    (diagonal-math MKL/vdFloor ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fceil [_ a y]
    (diagonal-math MKL/vdCeil ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (trunc [_ a y]
    (diagonal-math MKL/vdTrunc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (round [_ a y]
    (diagonal-math MKL/vdRound ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (modf [_ a y z]
    (diagonal-math MKL/vdModf ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (frac [_ a y]
    (diagonal-math MKL/vdFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fmin [_ a b y]
    (diagonal-math MKL/vdFmin ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (fmax [_ a b y]
    (diagonal-math MKL/vdFmax ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (copy-sign [_ a b y]
    (diagonal-math MKL/vdCopySign ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype FloatGTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/sswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/scopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/sscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/sdot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/snrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/sasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/saxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/slagtm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/slagtm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/ssum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/slaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/saxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/slasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a ipiv]
    (diagonal-trf LAPACK/sgttrf ^RealDiagonalMatrix a ^IntegerBlockVector ipiv))
  (trf [_ _]
    (dragan-says-ex "Pivotless factorization is not available for GT matrices."))
  (tri [_ _ _]
    (dragan-says-ex "Inverse is not available for GT matrices."))
  (trs [_ lu b ipiv]
    (gt-trs LAPACK/sgttrs ^RealDiagonalMatrix lu ^RealGEMatrix b ^IntegerBlockVector ipiv))
  (sv [_ a b pure]
    (gt-sv LAPACK/sgtsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (gt-con LAPACK/sgtcon ^RealDiagonalMatrix lu ^IntegerBlockVector ipiv nrm nrm1?))
  VectorMath
  (sqr [_ a y]
    (diagonal-math MKL/vsSqr ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (mul [_ a b y]
    (diagonal-math MKL/vsMul ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (div [_ a b y]
    (diagonal-math MKL/vsDiv ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (inv [_ a y]
    (diagonal-math MKL/vsInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (abs [_ a y]
    (diagonal-math MKL/vsAbs ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (diagonal-linear-frac MKL/vsLinearFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix b
                          scalea shifta scaleb shiftb ^RealDiagonalMatrix y))
  (fmod [_ a b y]
    (diagonal-math MKL/vsFmod ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (frem [_ a b y]
    (diagonal-math MKL/vsRemainder ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sqrt [_ a y]
    (diagonal-math MKL/vsSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-sqrt [_ a y]
    (diagonal-math MKL/vsInvSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cbrt [_ a y]
    (diagonal-math MKL/vsCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-cbrt [_ a y]
    (diagonal-math MKL/vsInvCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow2o3 [_ a y]
    (diagonal-math MKL/vsPow2o3 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow3o2 [_ a y]
    (diagonal-math MKL/vsPow3o2 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow [_ a b y]
    (diagonal-math MKL/vsPow ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (powx [_ a b y]
    (diagonal-powx MKL/vsPowx ^RealDiagonalMatrix a b ^RealDiagonalMatrix y))
  (hypot [_ a b y]
    (diagonal-math MKL/vsHypot ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (exp [_ a y]
    (diagonal-math MKL/vsExp ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expm1 [_ a y]
    (diagonal-math MKL/vsExpm1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log [_ a y]
    (diagonal-math MKL/vsLn ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log10 [_ a y]
    (diagonal-math MKL/vsLog10 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sin [_ a y]
    (diagonal-math MKL/vsSin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cos [_ a y]
    (diagonal-math MKL/vsCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tan [_ a y]
    (diagonal-math MKL/vsTan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sincos [_ a y z]
    (diagonal-math MKL/vsSinCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (asin [_ a y]
    (diagonal-math MKL/vsAsin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acos [_ a y]
    (diagonal-math MKL/vsAcos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan [_ a y]
    (diagonal-math MKL/vsAtan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan2 [_ a b y]
    (diagonal-math MKL/vsAtan2 ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sinh [_ a y]
    (diagonal-math MKL/vsSinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cosh [_ a y]
    (diagonal-math MKL/vsCosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tanh [_ a y]
    (diagonal-math MKL/vsTanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (asinh [_ a y]
    (diagonal-math MKL/vsAsinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acosh [_ a y]
    (diagonal-math MKL/vsAcosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atanh [_ a y]
    (diagonal-math MKL/vsAtanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf [_ a y]
    (diagonal-math MKL/vsErf ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc [_ a y]
    (diagonal-math MKL/vsErfc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf-inv [_ a y]
    (diagonal-math MKL/vsErfInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc-inv [_ a y]
    (diagonal-math MKL/vsErfcInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm [_ a y]
    (diagonal-math MKL/vsCdfNorm ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm-inv [_ a y]
    (diagonal-math MKL/vsCdfNormInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (gamma [_ a y]
    (diagonal-math MKL/vsGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (lgamma [_ a y]
    (diagonal-math MKL/vsLGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expint1 [_ a y]
    (diagonal-math MKL/vsExpInt1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (floor [_ a y]
    (diagonal-math MKL/vsFloor ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fceil [_ a y]
    (diagonal-math MKL/vsCeil ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (trunc [_ a y]
    (diagonal-math MKL/vsTrunc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (round [_ a y]
    (diagonal-math MKL/vsRound ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (modf [_ a y z]
    (diagonal-math MKL/vsModf ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (frac [_ a y]
    (diagonal-math MKL/vsFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fmin [_ a b y]
    (diagonal-math MKL/vsFmin ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (fmax [_ a b y]
    (diagonal-math MKL/vsFmax ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (copy-sign [_ a b y]
    (diagonal-math MKL/vsCopySign ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype DoubleGDEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/dswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/dcopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/dscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/ddot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/dnrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/dasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/daxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (gd-mv CBLAS/dsbmv alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (gd-mv LAPACK/dlascl2 ^RealDiagonalMatrix a ^RealBlockVector x))
  (mm [_ alpha a b beta c left]
    (gd-mm CBLAS/dsbmv alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ alpha a b left]
    (gd-mm LAPACK/dlascl2 CBLAS/dtbmv alpha ^RealDiagonalMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/dsum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/dlaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/daxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/dlasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (tri [_ a]
    (gd-tri MKL/vdInv ^RealDiagonalMatrix a))
  (trs [_ a b]
    (gd-trs LAPACK/dtbtrs ^RealDiagonalMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (gd-sv MKL/vdDiv CBLAS/dtbsv ^RealDiagonalMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (gd-con LAPACK/dtbcon ^RealDiagonalMatrix a nrm1?))
  VectorMath
  (sqr [_ a y]
    (diagonal-math MKL/vdSqr ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (mul [_ a b y]
    (diagonal-math MKL/vdMul ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (div [_ a b y]
    (diagonal-math MKL/vdDiv ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (inv [_ a y]
    (diagonal-math MKL/vdInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (abs [_ a y]
    (diagonal-math MKL/vdAbs ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (diagonal-linear-frac MKL/vdLinearFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix b
                          scalea shifta scaleb shiftb ^RealDiagonalMatrix y))
  (fmod [_ a b y]
    (diagonal-math MKL/vdFmod ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (frem [_ a b y]
    (diagonal-math MKL/vdRemainder ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sqrt [_ a y]
    (diagonal-math MKL/vdSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-sqrt [_ a y]
    (diagonal-math MKL/vdInvSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cbrt [_ a y]
    (diagonal-math MKL/vdCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-cbrt [_ a y]
    (diagonal-math MKL/vdInvCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow2o3 [_ a y]
    (diagonal-math MKL/vdPow2o3 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow3o2 [_ a y]
    (diagonal-math MKL/vdPow3o2 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow [_ a b y]
    (diagonal-math MKL/vdPow ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (powx [_ a b y]
    (diagonal-powx MKL/vdPowx ^RealDiagonalMatrix a b ^RealDiagonalMatrix y))
  (hypot [_ a b y]
    (diagonal-math MKL/vdHypot ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (exp [_ a y]
    (diagonal-math MKL/vdExp ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expm1 [_ a y]
    (diagonal-math MKL/vdExpm1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log [_ a y]
    (diagonal-math MKL/vdLn ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log10 [_ a y]
    (diagonal-math MKL/vdLog10 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sin [_ a y]
    (diagonal-math MKL/vdSin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cos [_ a y]
    (diagonal-math MKL/vdCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tan [_ a y]
    (diagonal-math MKL/vdTan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sincos [_ a y z]
    (diagonal-math MKL/vdSinCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (asin [_ a y]
    (diagonal-math MKL/vdAsin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acos [_ a y]
    (diagonal-math MKL/vdAcos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan [_ a y]
    (diagonal-math MKL/vdAtan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan2 [_ a b y]
    (diagonal-math MKL/vdAtan2 ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sinh [_ a y]
    (diagonal-math MKL/vdSinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cosh [_ a y]
    (diagonal-math MKL/vdCosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tanh [_ a y]
    (diagonal-math MKL/vdTanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (asinh [_ a y]
    (diagonal-math MKL/vdAsinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acosh [_ a y]
    (diagonal-math MKL/vdAcosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atanh [_ a y]
    (diagonal-math MKL/vdAtanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf [_ a y]
    (diagonal-math MKL/vdErf ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc [_ a y]
    (diagonal-math MKL/vdErfc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf-inv [_ a y]
    (diagonal-math MKL/vdErfInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc-inv [_ a y]
    (diagonal-math MKL/vdErfcInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm [_ a y]
    (diagonal-math MKL/vdCdfNorm ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm-inv [_ a y]
    (diagonal-math MKL/vdCdfNormInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (gamma [_ a y]
    (diagonal-math MKL/vdGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (lgamma [_ a y]
    (diagonal-math MKL/vdLGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expint1 [_ a y]
    (diagonal-math MKL/vdExpInt1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (floor [_ a y]
    (diagonal-math MKL/vdFloor ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fceil [_ a y]
    (diagonal-math MKL/vdCeil ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (trunc [_ a y]
    (diagonal-math MKL/vdTrunc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (round [_ a y]
    (diagonal-math MKL/vdRound ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (modf [_ a y z]
    (diagonal-math MKL/vdModf ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (frac [_ a y]
    (diagonal-math MKL/vdFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fmin [_ a b y]
    (diagonal-math MKL/vdFmin ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (fmax [_ a b y]
    (diagonal-math MKL/vdFmax ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (copy-sign [_ a b y]
    (diagonal-math MKL/vdCopySign ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype FloatGDEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/sswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/scopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/sscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/sdot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/snrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/sasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/saxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (gd-mv CBLAS/ssbmv alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a x]
    (gd-mv LAPACK/slascl2 ^RealDiagonalMatrix a ^RealBlockVector x))
  (mm [_ alpha a b beta c left]
    (gd-mm CBLAS/ssbmv alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ alpha a b left]
    (gd-mm LAPACK/slascl2 CBLAS/stbmv alpha ^RealDiagonalMatrix a ^RealGEMatrix b left))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/ssum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/slaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/saxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/slasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (tri [_ a]
    (gd-tri MKL/vsInv ^RealDiagonalMatrix a))
  (trs [_ a b]
    (gd-trs LAPACK/stbtrs ^RealDiagonalMatrix a ^RealGEMatrix b))
  (sv [_ a b _]
    (gd-sv MKL/vsDiv CBLAS/stbsv ^RealDiagonalMatrix a ^RealGEMatrix b))
  (con [_ a nrm1?]
    (gd-con LAPACK/stbcon ^RealDiagonalMatrix a nrm1?))
  VectorMath
  (sqr [_ a y]
    (diagonal-math MKL/vsSqr ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (mul [_ a b y]
    (diagonal-math MKL/vsMul ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (div [_ a b y]
    (diagonal-math MKL/vsDiv ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (inv [_ a y]
    (diagonal-math MKL/vsInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (abs [_ a y]
    (diagonal-math MKL/vsAbs ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (diagonal-linear-frac MKL/vsLinearFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix b
                          scalea shifta scaleb shiftb ^RealDiagonalMatrix y))
  (fmod [_ a b y]
    (diagonal-math MKL/vsFmod ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (frem [_ a b y]
    (diagonal-math MKL/vsRemainder ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sqrt [_ a y]
    (diagonal-math MKL/vsSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-sqrt [_ a y]
    (diagonal-math MKL/vsInvSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cbrt [_ a y]
    (diagonal-math MKL/vsCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-cbrt [_ a y]
    (diagonal-math MKL/vsInvCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow2o3 [_ a y]
    (diagonal-math MKL/vsPow2o3 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow3o2 [_ a y]
    (diagonal-math MKL/vsPow3o2 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow [_ a b y]
    (diagonal-math MKL/vsPow ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (powx [_ a b y]
    (diagonal-powx MKL/vsPowx ^RealDiagonalMatrix a b ^RealDiagonalMatrix y))
  (hypot [_ a b y]
    (diagonal-math MKL/vsHypot ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (exp [_ a y]
    (diagonal-math MKL/vsExp ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expm1 [_ a y]
    (diagonal-math MKL/vsExpm1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log [_ a y]
    (diagonal-math MKL/vsLn ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log10 [_ a y]
    (diagonal-math MKL/vsLog10 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sin [_ a y]
    (diagonal-math MKL/vsSin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cos [_ a y]
    (diagonal-math MKL/vsCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tan [_ a y]
    (diagonal-math MKL/vsTan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sincos [_ a y z]
    (diagonal-math MKL/vsSinCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (asin [_ a y]
    (diagonal-math MKL/vsAsin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acos [_ a y]
    (diagonal-math MKL/vsAcos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan [_ a y]
    (diagonal-math MKL/vsAtan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan2 [_ a b y]
    (diagonal-math MKL/vsAtan2 ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sinh [_ a y]
    (diagonal-math MKL/vsSinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cosh [_ a y]
    (diagonal-math MKL/vsCosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tanh [_ a y]
    (diagonal-math MKL/vsTanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (asinh [_ a y]
    (diagonal-math MKL/vsAsinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acosh [_ a y]
    (diagonal-math MKL/vsAcosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atanh [_ a y]
    (diagonal-math MKL/vsAtanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf [_ a y]
    (diagonal-math MKL/vsErf ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc [_ a y]
    (diagonal-math MKL/vsErfc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf-inv [_ a y]
    (diagonal-math MKL/vsErfInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc-inv [_ a y]
    (diagonal-math MKL/vsErfcInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm [_ a y]
    (diagonal-math MKL/vsCdfNorm ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm-inv [_ a y]
    (diagonal-math MKL/vsCdfNormInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (gamma [_ a y]
    (diagonal-math MKL/vsGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (lgamma [_ a y]
    (diagonal-math MKL/vsLGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expint1 [_ a y]
    (diagonal-math MKL/vsExpInt1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (floor [_ a y]
    (diagonal-math MKL/vsFloor ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fceil [_ a y]
    (diagonal-math MKL/vsCeil ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (trunc [_ a y]
    (diagonal-math MKL/vsTrunc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (round [_ a y]
    (diagonal-math MKL/vsRound ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (modf [_ a y z]
    (diagonal-math MKL/vsModf ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (frac [_ a y]
    (diagonal-math MKL/vsFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fmin [_ a b y]
    (diagonal-math MKL/vsFmin ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (fmax [_ a b y]
    (diagonal-math MKL/vsFmax ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (copy-sign [_ a b y]
    (diagonal-math MKL/vsCopySign ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype DoubleDTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/dswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/dcopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/dscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/ddot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/dnrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/dlangt (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/dasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/daxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/dlagtm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/dlagtm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/dsum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/dlaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/daxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/dlasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a]
    (diagonal-trf LAPACK/ddttrfb ^RealDiagonalMatrix a))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for DT matrices."))
  (trs [_ lu b]
    (dt-trs LAPACK/ddttrsb ^RealDiagonalMatrix lu ^RealGEMatrix b))
  (sv [_ a b pure]
    (dt-sv LAPACK/ddtsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (dragan-says-ex "Condition number is not available for DT matrices."))
  VectorMath
  (sqr [_ a y]
    (diagonal-math MKL/vdSqr ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (mul [_ a b y]
    (diagonal-math MKL/vdMul ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (div [_ a b y]
    (diagonal-math MKL/vdDiv ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (inv [_ a y]
    (diagonal-math MKL/vdInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (abs [_ a y]
    (diagonal-math MKL/vdAbs ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (diagonal-linear-frac MKL/vdLinearFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix b
                          scalea shifta scaleb shiftb ^RealDiagonalMatrix y))
  (fmod [_ a b y]
    (diagonal-math MKL/vdFmod ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (frem [_ a b y]
    (diagonal-math MKL/vdRemainder ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sqrt [_ a y]
    (diagonal-math MKL/vdSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-sqrt [_ a y]
    (diagonal-math MKL/vdInvSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cbrt [_ a y]
    (diagonal-math MKL/vdCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-cbrt [_ a y]
    (diagonal-math MKL/vdInvCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow2o3 [_ a y]
    (diagonal-math MKL/vdPow2o3 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow3o2 [_ a y]
    (diagonal-math MKL/vdPow3o2 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow [_ a b y]
    (diagonal-math MKL/vdPow ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (powx [_ a b y]
    (diagonal-powx MKL/vdPowx ^RealDiagonalMatrix a b ^RealDiagonalMatrix y))
  (hypot [_ a b y]
    (diagonal-math MKL/vdHypot ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (exp [_ a y]
    (diagonal-math MKL/vdExp ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expm1 [_ a y]
    (diagonal-math MKL/vdExpm1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log [_ a y]
    (diagonal-math MKL/vdLn ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log10 [_ a y]
    (diagonal-math MKL/vdLog10 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sin [_ a y]
    (diagonal-math MKL/vdSin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cos [_ a y]
    (diagonal-math MKL/vdCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tan [_ a y]
    (diagonal-math MKL/vdTan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sincos [_ a y z]
    (diagonal-math MKL/vdSinCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (asin [_ a y]
    (diagonal-math MKL/vdAsin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acos [_ a y]
    (diagonal-math MKL/vdAcos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan [_ a y]
    (diagonal-math MKL/vdAtan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan2 [_ a b y]
    (diagonal-math MKL/vdAtan2 ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sinh [_ a y]
    (diagonal-math MKL/vdSinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cosh [_ a y]
    (diagonal-math MKL/vdCosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tanh [_ a y]
    (diagonal-math MKL/vdTanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (asinh [_ a y]
    (diagonal-math MKL/vdAsinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acosh [_ a y]
    (diagonal-math MKL/vdAcosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atanh [_ a y]
    (diagonal-math MKL/vdAtanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf [_ a y]
    (diagonal-math MKL/vdErf ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc [_ a y]
    (diagonal-math MKL/vdErfc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf-inv [_ a y]
    (diagonal-math MKL/vdErfInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc-inv [_ a y]
    (diagonal-math MKL/vdErfcInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm [_ a y]
    (diagonal-math MKL/vdCdfNorm ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm-inv [_ a y]
    (diagonal-math MKL/vdCdfNormInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (gamma [_ a y]
    (diagonal-math MKL/vdGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (lgamma [_ a y]
    (diagonal-math MKL/vdLGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expint1 [_ a y]
    (diagonal-math MKL/vdExpInt1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (floor [_ a y]
    (diagonal-math MKL/vdFloor ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fceil [_ a y]
    (diagonal-math MKL/vdCeil ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (trunc [_ a y]
    (diagonal-math MKL/vdTrunc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (round [_ a y]
    (diagonal-math MKL/vdRound ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (modf [_ a y z]
    (diagonal-math MKL/vdModf ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (frac [_ a y]
    (diagonal-math MKL/vdFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fmin [_ a b y]
    (diagonal-math MKL/vdFmin ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (fmax [_ a b y]
    (diagonal-math MKL/vdFmax ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (copy-sign [_ a b y]
    (diagonal-math MKL/vdCopySign ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype FloatDTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/sswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/scopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/sscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (diagonal-method CBLAS/sdot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/slangt (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (diagonal-method CBLAS/snrm2 ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/slangt (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (diagonal-method CBLAS/sasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/saxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/slagtm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/slagtm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (diagonal-method CBLAS/ssum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/slaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/saxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/slasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a]
    (diagonal-trf LAPACK/sdttrfb ^RealDiagonalMatrix a))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for DT matrices."))
  (trs [_ lu b]
    (dt-trs LAPACK/sdttrsb ^RealDiagonalMatrix lu ^RealGEMatrix b))
  (sv [_ a b pure]
    (dt-sv LAPACK/sdtsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (dragan-says-ex "Condition number is not available for DT matrices."))
  VectorMath
  (sqr [_ a y]
    (diagonal-math MKL/vsSqr ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (mul [_ a b y]
    (diagonal-math MKL/vsMul ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (div [_ a b y]
    (diagonal-math MKL/vsDiv ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (inv [_ a y]
    (diagonal-math MKL/vsInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (abs [_ a y]
    (diagonal-math MKL/vsAbs ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (diagonal-linear-frac MKL/vsLinearFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix b
                          scalea shifta scaleb shiftb ^RealDiagonalMatrix y))
  (fmod [_ a b y]
    (diagonal-math MKL/vsFmod ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (frem [_ a b y]
    (diagonal-math MKL/vsRemainder ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sqrt [_ a y]
    (diagonal-math MKL/vsSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-sqrt [_ a y]
    (diagonal-math MKL/vsInvSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cbrt [_ a y]
    (diagonal-math MKL/vsCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-cbrt [_ a y]
    (diagonal-math MKL/vsInvCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow2o3 [_ a y]
    (diagonal-math MKL/vsPow2o3 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow3o2 [_ a y]
    (diagonal-math MKL/vsPow3o2 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow [_ a b y]
    (diagonal-math MKL/vsPow ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (powx [_ a b y]
    (diagonal-powx MKL/vsPowx ^RealDiagonalMatrix a b ^RealDiagonalMatrix y))
  (hypot [_ a b y]
    (diagonal-math MKL/vsHypot ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (exp [_ a y]
    (diagonal-math MKL/vsExp ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expm1 [_ a y]
    (diagonal-math MKL/vsExpm1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log [_ a y]
    (diagonal-math MKL/vsLn ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log10 [_ a y]
    (diagonal-math MKL/vsLog10 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sin [_ a y]
    (diagonal-math MKL/vsSin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cos [_ a y]
    (diagonal-math MKL/vsCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tan [_ a y]
    (diagonal-math MKL/vsTan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sincos [_ a y z]
    (diagonal-math MKL/vsSinCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (asin [_ a y]
    (diagonal-math MKL/vsAsin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acos [_ a y]
    (diagonal-math MKL/vsAcos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan [_ a y]
    (diagonal-math MKL/vsAtan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan2 [_ a b y]
    (diagonal-math MKL/vsAtan2 ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sinh [_ a y]
    (diagonal-math MKL/vsSinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cosh [_ a y]
    (diagonal-math MKL/vsCosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tanh [_ a y]
    (diagonal-math MKL/vsTanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (asinh [_ a y]
    (diagonal-math MKL/vsAsinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acosh [_ a y]
    (diagonal-math MKL/vsAcosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atanh [_ a y]
    (diagonal-math MKL/vsAtanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf [_ a y]
    (diagonal-math MKL/vsErf ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc [_ a y]
    (diagonal-math MKL/vsErfc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf-inv [_ a y]
    (diagonal-math MKL/vsErfInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc-inv [_ a y]
    (diagonal-math MKL/vsErfcInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm [_ a y]
    (diagonal-math MKL/vsCdfNorm ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm-inv [_ a y]
    (diagonal-math MKL/vsCdfNormInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (gamma [_ a y]
    (diagonal-math MKL/vsGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (lgamma [_ a y]
    (diagonal-math MKL/vsLGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expint1 [_ a y]
    (diagonal-math MKL/vsExpInt1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (floor [_ a y]
    (diagonal-math MKL/vsFloor ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fceil [_ a y]
    (diagonal-math MKL/vsCeil ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (trunc [_ a y]
    (diagonal-math MKL/vsTrunc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (round [_ a y]
    (diagonal-math MKL/vsRound ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (modf [_ a y z]
    (diagonal-math MKL/vsModf ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (frac [_ a y]
    (diagonal-math MKL/vsFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fmin [_ a b y]
    (diagonal-math MKL/vsFmin ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (fmax [_ a b y]
    (diagonal-math MKL/vsFmax ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (copy-sign [_ a b y]
    (diagonal-math MKL/vsCopySign ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype DoubleSTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/dswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/dcopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/dscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (st-sum CBLAS/ddot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/dlanst (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (tridiagonal-lan LAPACK/dlanst (int \F) ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/dlanst (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (st-sum CBLAS/dasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/daxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/dlastm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/dlastm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/idamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (st-sum CBLAS/dsum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/dlaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/daxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/dlasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a]
    (diagonal-trf LAPACK/dpttrf ^RealDiagonalMatrix a))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for ST matrices."))
  (trs [_ lu b]
    (st-trs LAPACK/dpttrs ^RealDiagonalMatrix lu ^RealGEMatrix b))
  (sv [_ a b pure]
    (st-sv LAPACK/dptsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (dragan-says-ex "Condition number is not available for ST matrices."))
  VectorMath
  (sqr [_ a y]
    (diagonal-math MKL/vdSqr ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (mul [_ a b y]
    (diagonal-math MKL/vdMul ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (div [_ a b y]
    (diagonal-math MKL/vdDiv ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (inv [_ a y]
    (diagonal-math MKL/vdInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (abs [_ a y]
    (diagonal-math MKL/vdAbs ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (diagonal-linear-frac MKL/vdLinearFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix b
                          scalea shifta scaleb shiftb ^RealDiagonalMatrix y))
  (fmod [_ a b y]
    (diagonal-math MKL/vdFmod ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (frem [_ a b y]
    (diagonal-math MKL/vdRemainder ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sqrt [_ a y]
    (diagonal-math MKL/vdSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-sqrt [_ a y]
    (diagonal-math MKL/vdInvSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cbrt [_ a y]
    (diagonal-math MKL/vdCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-cbrt [_ a y]
    (diagonal-math MKL/vdInvCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow2o3 [_ a y]
    (diagonal-math MKL/vdPow2o3 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow3o2 [_ a y]
    (diagonal-math MKL/vdPow3o2 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow [_ a b y]
    (diagonal-math MKL/vdPow ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (powx [_ a b y]
    (diagonal-powx MKL/vdPowx ^RealDiagonalMatrix a b ^RealDiagonalMatrix y))
  (hypot [_ a b y]
    (diagonal-math MKL/vdHypot ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (exp [_ a y]
    (diagonal-math MKL/vdExp ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expm1 [_ a y]
    (diagonal-math MKL/vdExpm1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log [_ a y]
    (diagonal-math MKL/vdLn ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log10 [_ a y]
    (diagonal-math MKL/vdLog10 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sin [_ a y]
    (diagonal-math MKL/vdSin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cos [_ a y]
    (diagonal-math MKL/vdCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tan [_ a y]
    (diagonal-math MKL/vdTan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sincos [_ a y z]
    (diagonal-math MKL/vdSinCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (asin [_ a y]
    (diagonal-math MKL/vdAsin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acos [_ a y]
    (diagonal-math MKL/vdAcos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan [_ a y]
    (diagonal-math MKL/vdAtan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan2 [_ a b y]
    (diagonal-math MKL/vdAtan2 ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sinh [_ a y]
    (diagonal-math MKL/vdSinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cosh [_ a y]
    (diagonal-math MKL/vdCosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tanh [_ a y]
    (diagonal-math MKL/vdTanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (asinh [_ a y]
    (diagonal-math MKL/vdAsinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acosh [_ a y]
    (diagonal-math MKL/vdAcosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atanh [_ a y]
    (diagonal-math MKL/vdAtanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf [_ a y]
    (diagonal-math MKL/vdErf ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc [_ a y]
    (diagonal-math MKL/vdErfc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf-inv [_ a y]
    (diagonal-math MKL/vdErfInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc-inv [_ a y]
    (diagonal-math MKL/vdErfcInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm [_ a y]
    (diagonal-math MKL/vdCdfNorm ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm-inv [_ a y]
    (diagonal-math MKL/vdCdfNormInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (gamma [_ a y]
    (diagonal-math MKL/vdGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (lgamma [_ a y]
    (diagonal-math MKL/vdLGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expint1 [_ a y]
    (diagonal-math MKL/vdExpInt1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (floor [_ a y]
    (diagonal-math MKL/vdFloor ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fceil [_ a y]
    (diagonal-math MKL/vdCeil ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (trunc [_ a y]
    (diagonal-math MKL/vdTrunc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (round [_ a y]
    (diagonal-math MKL/vdRound ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (modf [_ a y z]
    (diagonal-math MKL/vdModf ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (frac [_ a y]
    (diagonal-math MKL/vdFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fmin [_ a b y]
    (diagonal-math MKL/vdFmin ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (fmax [_ a b y]
    (diagonal-math MKL/vdFmax ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (copy-sign [_ a b y]
    (diagonal-math MKL/vdCopySign ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

(deftype FloatSTEngine []
  Blas
  (swap [_ a b]
    (diagonal-method CBLAS/sswap ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    a)
  (copy [_ a b]
    (diagonal-method CBLAS/scopy ^RealDiagonalMatrix a ^RealDiagonalMatrix b)
    b)
  (scal [_ alpha a]
    (diagonal-scal CBLAS/sscal alpha ^RealDiagonalMatrix a))
  (dot [_ a b]
    (st-sum CBLAS/sdot ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (nrm1 [_ a]
    (tridiagonal-lan LAPACK/slanst (int \O) ^RealDiagonalMatrix a))
  (nrm2 [_ a]
    (tridiagonal-lan LAPACK/slanst (int \F) ^RealDiagonalMatrix a))
  (nrmi [_ a]
    (tridiagonal-lan LAPACK/slanst (int \I) ^RealDiagonalMatrix a))
  (asum [_ a]
    (st-sum CBLAS/sasum ^RealDiagonalMatrix a))
  (axpy [_ alpha a b]
    (diagonal-axpy CBLAS/saxpy alpha ^RealDiagonalMatrix a ^RealDiagonalMatrix b))
  (mv [_ alpha a x beta y]
    (tridiagonal-mv LAPACK/slastm alpha ^RealDiagonalMatrix a ^RealBlockVector x beta ^RealBlockVector y))
  (mv [_ a _]
    (tridiagonal-mv a))
  (mm [_ alpha a b beta c left]
    (tridiagonal-mm LAPACK/slastm alpha ^RealDiagonalMatrix a ^RealGEMatrix b beta ^RealGEMatrix c left))
  (mm [_ _ a _ _]
    (tridiagonal-mm a))
  BlasPlus
  (amax [_ a]
    (diagonal-amax CBLAS/isamax ^RealDiagonalMatrix a))
  (sum [_ a]
    (st-sum CBLAS/ssum ^RealDiagonalMatrix a))
  (set-all [_ alpha a]
    (diagonal-laset LAPACK/slaset alpha ^RealDiagonalMatrix a))
  (axpby [_ alpha a beta b]
    (diagonal-axpby MKL/saxpby alpha ^RealDiagonalMatrix a beta ^RealDiagonalMatrix b))
  Lapack
  (srt [_ a increasing]
    (diagonal-lasrt LAPACK/slasrt ^RealDiagonalMatrix a increasing))
  (laswp [_ _ _ _ _]
    (dragan-says-ex "Pivoted swap is not available for diagonal matrices."))
  (trf [_ a]
    (diagonal-trf LAPACK/spttrf ^RealDiagonalMatrix a))
  (tri [_ _]
    (dragan-says-ex "Inverse is not available for ST matrices."))
  (trs [_ lu b]
    (st-trs LAPACK/spttrs ^RealDiagonalMatrix lu ^RealGEMatrix b))
  (sv [_ a b pure]
    (st-sv LAPACK/sptsv ^RealDiagonalMatrix a ^RealGEMatrix b pure))
  (con [_ lu ipiv nrm nrm1?]
    (dragan-says-ex "Condition number is not available for ST matrices."))
  VectorMath
  (sqr [_ a y]
    (diagonal-math MKL/vsSqr ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (mul [_ a b y]
    (diagonal-math MKL/vsMul ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (div [_ a b y]
    (diagonal-math MKL/vsDiv ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (inv [_ a y]
    (diagonal-math MKL/vsInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (abs [_ a y]
    (diagonal-math MKL/vsAbs ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (linear-frac [_ a b scalea shifta scaleb shiftb y]
    (diagonal-linear-frac MKL/vsLinearFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix b
                          scalea shifta scaleb shiftb ^RealDiagonalMatrix y))
  (fmod [_ a b y]
    (diagonal-math MKL/vsFmod ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (frem [_ a b y]
    (diagonal-math MKL/vsRemainder ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sqrt [_ a y]
    (diagonal-math MKL/vsSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-sqrt [_ a y]
    (diagonal-math MKL/vsInvSqrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cbrt [_ a y]
    (diagonal-math MKL/vsCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (inv-cbrt [_ a y]
    (diagonal-math MKL/vsInvCbrt ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow2o3 [_ a y]
    (diagonal-math MKL/vsPow2o3 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow3o2 [_ a y]
    (diagonal-math MKL/vsPow3o2 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (pow [_ a b y]
    (diagonal-math MKL/vsPow ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (powx [_ a b y]
    (diagonal-powx MKL/vsPowx ^RealDiagonalMatrix a b ^RealDiagonalMatrix y))
  (hypot [_ a b y]
    (diagonal-math MKL/vsHypot ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (exp [_ a y]
    (diagonal-math MKL/vsExp ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expm1 [_ a y]
    (diagonal-math MKL/vsExpm1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log [_ a y]
    (diagonal-math MKL/vsLn ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (log10 [_ a y]
    (diagonal-math MKL/vsLog10 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sin [_ a y]
    (diagonal-math MKL/vsSin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cos [_ a y]
    (diagonal-math MKL/vsCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tan [_ a y]
    (diagonal-math MKL/vsTan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (sincos [_ a y z]
    (diagonal-math MKL/vsSinCos ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (asin [_ a y]
    (diagonal-math MKL/vsAsin ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acos [_ a y]
    (diagonal-math MKL/vsAcos ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan [_ a y]
    (diagonal-math MKL/vsAtan ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atan2 [_ a b y]
    (diagonal-math MKL/vsAtan2 ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sinh [_ a y]
    (diagonal-math MKL/vsSinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cosh [_ a y]
    (diagonal-math MKL/vsCosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (tanh [_ a y]
    (diagonal-math MKL/vsTanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (asinh [_ a y]
    (diagonal-math MKL/vsAsinh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (acosh [_ a y]
    (diagonal-math MKL/vsAcosh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (atanh [_ a y]
    (diagonal-math MKL/vsAtanh ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf [_ a y]
    (diagonal-math MKL/vsErf ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc [_ a y]
    (diagonal-math MKL/vsErfc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erf-inv [_ a y]
    (diagonal-math MKL/vsErfInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (erfc-inv [_ a y]
    (diagonal-math MKL/vsErfcInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm [_ a y]
    (diagonal-math MKL/vsCdfNorm ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (cdf-norm-inv [_ a y]
    (diagonal-math MKL/vsCdfNormInv ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (gamma [_ a y]
    (diagonal-math MKL/vsGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (lgamma [_ a y]
    (diagonal-math MKL/vsLGamma ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (expint1 [_ a y]
    (diagonal-math MKL/vsExpInt1 ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (floor [_ a y]
    (diagonal-math MKL/vsFloor ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fceil [_ a y]
    (diagonal-math MKL/vsCeil ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (trunc [_ a y]
    (diagonal-math MKL/vsTrunc ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (round [_ a y]
    (diagonal-math MKL/vsRound ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (modf [_ a y z]
    (diagonal-math MKL/vsModf ^RealDiagonalMatrix a ^RealDiagonalMatrix y ^RealDiagonalMatrix z))
  (frac [_ a y]
    (diagonal-math MKL/vsFrac ^RealDiagonalMatrix a ^RealDiagonalMatrix y))
  (fmin [_ a b y]
    (diagonal-math MKL/vsFmin ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (fmax [_ a b y]
    (diagonal-math MKL/vsFmax ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (copy-sign [_ a b y]
    (diagonal-math MKL/vsCopySign ^RealDiagonalMatrix a ^RealDiagonalMatrix b ^RealDiagonalMatrix y))
  (sigmoid [this a y]
    (sigmoid-over-tanh this a y))
  (ramp [this a y]
    (matrix-ramp this a y))
  (relu [this alpha a y]
    (matrix-relu this alpha a y))
  (elu [this alpha a y]
    (matrix-elu this alpha a y)))

;; =============== Factories ==================================================

(deftype MKLRealFactory [index-fact ^DataAccessor da
                         vector-eng ge-eng tr-eng sy-eng gb-eng sb-eng tb-eng sp-eng tp-eng
                         gd-eng gt-eng dt-eng st-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    @index-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  RngStreamFactory
  (create-rng-state [_ seed]
    (create-stream-ars5 seed))
  Factory
  (create-vector [this master buf n ofst strd]
    (real-block-vector this master buf n ofst strd))
  (create-vector [this n _]
    (real-block-vector this n))
  (create-ge [this m n column? _]
    (real-ge-matrix this m n column?))
  (create-uplo [this n mat-type column? lower? diag-unit? _]
    (real-uplo-matrix this n column? lower? diag-unit? mat-type))
  (create-tr [this n column? lower? diag-unit? _]
    (real-uplo-matrix this n column? lower? diag-unit?))
  (create-sy [this n column? lower? _]
    (real-uplo-matrix this n column? lower?))
  (create-banded [this m n kl ku matrix-type column? _]
    (real-banded-matrix this m n kl ku column? matrix-type))
  (create-gb [this m n kl ku lower? _]
    (real-banded-matrix this m n kl ku lower?))
  (create-tb [this n k column? lower? diag-unit? _]
    (if (or (and column? lower?) (and (not column?) (not lower?)))
      (real-tb-matrix this n k column? lower? diag-unit?)
      (dragan-says-ex "TB matrices have to be either column-major lower or row-major upper."
                      {:layout (if column? :column :row) :uplo (if lower? :lower :upper)})))
  (create-sb [this n k column? lower? _]
    (if (or (and column? lower?) (and (not column?) (not lower?)))
      (real-sb-matrix this n k column? lower?)
      (dragan-says-ex "SB matrices have to be either column-major lower or row-major upper."
                      {:layout (if column? :column :row) :uplo (if lower? :lower :upper)})))
  (create-packed [this n matrix-type column? lower? diag-unit? _]
    (real-packed-matrix this n column? lower? diag-unit? matrix-type))
  (create-tp [this n column? lower? diag-unit? _]
    (real-packed-matrix this n column? lower? diag-unit?))
  (create-sp [this n column? lower? _]
    (real-packed-matrix this n column? lower?))
  (create-diagonal [this n matrix-type _]
    (real-diagonal-matrix this n matrix-type))
  (vector-engine [_]
    vector-eng)
  (ge-engine [_]
    ge-eng)
  (tr-engine [_]
    tr-eng)
  (sy-engine [_]
    sy-eng)
  (gb-engine [_]
    gb-eng)
  (sb-engine [_]
    sb-eng)
  (tb-engine [_]
    tb-eng)
  (tp-engine [_]
    tp-eng)
  (sp-engine [_]
    sp-eng)
  (gd-engine [_]
    gd-eng)
  (gt-engine [_]
    gt-eng)
  (dt-engine [_]
    dt-eng)
  (st-engine [_]
    st-eng))

(deftype MKLIntegerFactory [index-fact ^DataAccessor da vector-eng]
  DataAccessorProvider
  (data-accessor [_]
    da)
  FactoryProvider
  (factory [this]
    this)
  (native-factory [this]
    this)
  (index-factory [this]
    @index-fact)
  MemoryContext
  (compatible? [_ o]
    (compatible? da o))
  Factory
  (create-vector [this master buf n ofst strd]
    (integer-block-vector this master buf n ofst strd))
  (create-vector [this n _]
    (integer-block-vector this n))
  (vector-engine [_]
    vector-eng))

(let [index-fact (volatile! nil)]

  (def mkl-int (->MKLIntegerFactory index-fact int-accessor (->IntVectorEngine)))

  (def mkl-long (->MKLIntegerFactory index-fact long-accessor (->LongVectorEngine)))

  (def mkl-byte (->MKLIntegerFactory index-fact byte-accessor (->ByteVectorEngine)))

  (def mkl-short (->MKLIntegerFactory index-fact short-accessor (->ShortVectorEngine)))

  (def mkl-float
    (->MKLRealFactory index-fact float-accessor (->FloatVectorEngine)
                      (->FloatGEEngine) (->FloatTREngine) (->FloatSYEngine)
                      (->FloatGBEngine) (->FloatSBEngine) (->FloatTBEngine)
                      (->FloatSPEngine) (->FloatTPEngine)
                      (->FloatGDEngine) (->FloatGTEngine) (->FloatDTEngine) (->FloatSTEngine)))

  (def mkl-double
    (->MKLRealFactory index-fact double-accessor (->DoubleVectorEngine)
                      (->DoubleGEEngine) (->DoubleTREngine) (->DoubleSYEngine)
                      (->DoubleGBEngine) (->DoubleSBEngine) (->DoubleTBEngine)
                      (->DoubleSPEngine) (->DoubleTPEngine)
                      (->DoubleGDEngine) (->DoubleGTEngine) (->DoubleDTEngine) (->DoubleSTEngine)))

  (vreset! index-fact mkl-int))

(extend-buffer FloatBuffer mkl-float DirectFloatBufferU true)
(extend-buffer DoubleBuffer mkl-double DirectDoubleBufferU true)
(extend-buffer LongBuffer mkl-long DirectLongBufferU true)
(extend-buffer IntBuffer mkl-int DirectIntBufferU true)
