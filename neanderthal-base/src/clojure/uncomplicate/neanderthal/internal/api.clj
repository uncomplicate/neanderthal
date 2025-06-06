;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.api
  (:refer-clojure :exclude [abs])
  (:require [uncomplicate.commons.core :refer [Releaseable]]))

;; ================================ Default entry =================================

(definterface Default
  (realEntry ^double [nav stor da buf ^long ofst ^long i ^long j])
  (integerEntry ^long [nav stor da buf ^long ofst ^long i ^long j]))

;; ================================ Storage =======================================

(definterface RealLayoutFlipper
  (^double get [a ^long i ^long j])
  (set [a, ^long i ^long j ^double val]))

(definterface IntegerLayoutFlipper
  (^long get [a ^long i ^long j])
  (set [a, ^long i ^long j ^long val]))

(defprotocol Flippable
  (flip [this]))

(defprotocol Flipper
  (real-flipper ^RealLayoutFlipper [this])
  (integer-flipper ^IntegerLayoutFlipper [this]))

(defprotocol Navigable
  (region ^Region [this])
  (storage ^DenseStorage [this])
  (navigator ^LayoutNavigator [this]))

;; ====================== Computation engines ========================================

(defprotocol Blas
  (iamax [this x])
  (iamin [this x])
  (swap [this x y])
  (copy [this x y])
  (dot [this x y])
  (nrm1 [this x])
  (nrm2 [this x])
  (nrmi [this x])
  (asum [this x])
  (rot [this x y c s])
  (rotg [this abcs])
  (rotm [this x y params])
  (rotmg [this d1d2xy param])
  (scal [this alpha x])
  (axpy [this alpha x y])
  (mv [this alpha a x beta y] [this a x])
  (rk [this alpha x a] [this alpha x y a])
  (srk [this alpha a beta c])
  (mm [this alpha a b beta c left] [this alpha a b left] [this alpha a b]))

(defprotocol BlasPlus
  (amax [this x])
  (sum [this x])
  (imax [this x])
  (imin [this x])
  (subcopy [this x y kx lx ky])
  (set-all [this alpha x])
  (axpby [this alpha x beta y])
  (trans [this a]))

(defprotocol Lapack
  (srt [this x increasing])
  (laswp [this a x k1 k2])
  (lapmr [this a k forward])
  (lapmt [this a k forward])
  (trf [this a ipiv] [this a])
  (trfx [this a])
  (tri [this lu ipiv] [this a])
  (trs [this lu b ipiv] [this a b])
  (con [this lu ipiv nrm nrm1?] [this gg nrm nrm1?] [this a nrm1?])
  (det [this lu ipiv] [this a])
  (sv [this a b pure] [this a b])
  (svd [this a s superb] [this a s u vt superb])
  (sdd [this a s] [this a s u vt])
  (qrf [this a tau])
  (qrfp [this a tau])
  (qp3 [this a jpiv tau])
  (gqr [this a tau])
  (mqr [this a tau c left])
  (rqf [this a tau])
  (grq [this a tau])
  (mrq [this a tau c left])
  (lqf [this a tau])
  (glq [this a tau])
  (mlq [this a tau c left])
  (qlf [this a tau])
  (gql [this a tau])
  (mql [this a tau c left])
  (ls [this a b])
  (lse [this a b c d x])
  (gls [this a b d x y])
  (ev [this a w vl vr])
  (es [this a w vs])
  (evr [this a w vl vr]))

(defprotocol VectorMath
  (sqr [this a y])
  (mul [this a b y])
  (div [this a b y])
  (inv [this a y])
  (abs [this a y])
  (linear-frac [this a b scalea shifta scaleb shiftb y])
  (fmod [this a b y])
  (frem [this a b y])
  (sqrt [this a y])
  (inv-sqrt [this a y])
  (cbrt [this a y])
  (inv-cbrt [this a y])
  (pow2o3 [this a y])
  (pow3o2 [this a y])
  (pow [this a b y])
  (powx [this a b y])
  (hypot [this a b y])
  (exp [this a y])
  (exp2 [this a y])
  (exp10 [this a y])
  (expm1 [this a y])
  (log [this a y])
  (log2 [this a y])
  (log10 [this a y])
  (log1p [this a y])
  (cos [this a y])
  (sin [this a y])
  (sincos [this a y z])
  (tan [this a y])
  (acos [this a y])
  (asin [this a y])
  (atan [this a y])
  (atan2 [this a b y])
  (cosh [this a y])
  (sinh [this a y])
  (tanh [this a y])
  (acosh [this a y])
  (asinh [this a y])
  (atanh [this a y])
  (erf [this a y])
  (erfc [this a y])
  (erf-inv [this a y])
  (erfc-inv [this a y])
  (cdf-norm [this a y])
  (cdf-norm-inv [this a y])
  (gamma [this a y])
  (lgamma [this a y])
  (expint1 [this a y])
  (floor [this a y])
  (fceil [this a y])
  (trunc [this a y])
  (round [this a y])
  (modf [this a y z])
  (frac [this a y])
  (fmin [this a b y])
  (fmax [this a b y])
  (copy-sign [this a b y])
  (sigmoid [this a y])
  (ramp [this a y])
  (relu [this alpha a y])
  (elu [this alpha a y] [this a y]))

(defprotocol Triangularizable
  (create-trf [a pure])
  (create-ptrf [a]))

(defprotocol TRF
  (trtrs! [a b])
  (trtrs [a b])
  (trtri! [a])
  (trtri [a])
  (trcon [a nrm nrm1?] [a nrm1?])
  (trdet [a]))

(defprotocol Orthogonalizable
  (create-qrf [a pure])
  (create-rqf [a pure])
  (create-qlf [a pure])
  (create-lqf [a pure]))

(defprotocol ORF
  (org! [or])
  (org [or]))

(defprotocol BlockEngine
  (equals-block [_ cu-x cu-y]))

(defprotocol RandomNumberGenerator
  (rand-normal [this rng-state a b x])
  (rand-uniform [this rng-state a b x]))

(defprotocol Factory
  (create-vector [this n init] [this master buf n ofst strd])
  (create-ge [this m n column? init])
  (create-uplo [this n matrix-type column? lower? diag-unit? init])
  (create-tr [this n column? lower? diag-unit? init])
  (create-sy [this n column? lower? init])
  (create-banded [this m n kl ku matrix-type column? init])
  (create-gb [this m n kl ku column? init])
  (create-tb [this n k column? lower? diag-unit? init])
  (create-sb [this n k column? lower? init])
  (create-packed [this n matrix-type column? lower? diag-unit? init])
  (create-tp [this n column? lower? diag-unit? init])
  (create-sp [this n column? lower? init])
  (create-diagonal [this n matrix-type init])
  (vector-engine [this])
  (ge-engine [this])
  (tr-engine [this])
  (sy-engine [this])
  (gb-engine [this])
  (sb-engine [this])
  (tb-engine [this])
  (sp-engine [this])
  (tp-engine [this])
  (gd-engine [this])
  (gt-engine [this])
  (dt-engine [this])
  (st-engine [this]))

(defprotocol UnsafeFactory
  (create-vector* [this master buf n strd]))

(defprotocol RngStreamFactory
  (create-rng-state [this seed]))

(defprotocol EngineProvider
  (engine [this]))

(defprotocol FactoryProvider
  (factory [this])
  (native-factory [this])
  (index-factory [this]))

(defprotocol DataAccessorProvider
  (data-accessor ^DataAccessor [this]))

(defprotocol FlowProvider
  (flow [this]))

(defprotocol MemoryContext
  (compatible? [this other])
  (device [this])
  (fits? [this other])
  (fits-navigation? [this other]))

(defprotocol Container
  (raw [this] [this factory])
  (zero [this] [this factory])
  (host [this])
  (native [this]))

(defprotocol DenseContainer
  (view-ge [this] [this stride-mult] [this m n])
  (view-tr [this lower? diag-unit?])
  (view-sy [this lower?])
  (view-vctr [this] [this stride-mult]))

(defprotocol Subband
  (subband [this kl ku]))

;; ============ Sparse API =====================================================

(defprotocol SparseBlas
  (gthr [this y x]))

(defprotocol CompressedSparse
  (entries [this])
  (indices [this]))

(defprotocol CSR
  (columns [this])
  (indexb [this])
  (indexe [this]))

(defprotocol SparseFactory
  (create-ge-csr [this m n ind ind-b ind-e column? init] [this a b indices?])
  (create-tr-csr [this m n ind ind-b ind-e column? lower? diag-unit? init])
  (create-sy-csr [this m n ind ind-b ind-e column? lower? diag-unit? init])
  (cs-vector-engine [this])
  (csr-engine [this]))

;; ============ Realeaseable ===================================================

(defprotocol Destructor
  (destruct [this p]))

(extend-type clojure.lang.Sequential
  Releaseable
  (release [this]
    true)
  Container
  (raw [this fact]
    (let [e1 (first this)
          n (count this)]
      (if (sequential? e1)
        (create-ge (factory fact) (count e1) n true false)
        (create-vector (factory fact) n false))))
  (native [this]
    this))

(extend-type Object
  MemoryContext
  (compatible? [this o]
    (instance? (class this) o))
  DataAccessorProvider
  (data-accessor [_]
    nil)
  Container
  (native [this]
    this))

(extend-type nil
  MemoryContext
  (compatible? [this o]
    false)
  DataAccessorProvider
  (data-accessor [_]
    nil))

;; ============================================================================

(defn options-column? [options]
  (not (= :row (:layout options))))

(defn options-lower? [options]
  (not (= :upper (:uplo options))))

(defn options-diag-unit? [options]
  (= :unit (:diag options)))

(defn dec-property
  [^long code]
  (case code
    101 :row
    102 :column
    111 :no-trans
    112 :trans
    113 :conj-trans
    121 :upper
    122 :lower
    131 :non-unit
    132 :unit
    :unknown))

(defn enc-property [option]
  (case option
    :row 101
    :column 102
    :no-trans 111
    :trans 112
    :conj-trans 113
    :upper 121
    :lower 122
    :non-unit 131
    :unit 132
    (throw (ex-info "Invalid option." {:option option}))))

(defn enc-layout ^long [layout]
  (case layout
    :row 101
    :column 102
    101 101
    102 102
    (throw (ex-info "Invalid layout" {:layout layout}))))

(defn enc-uplo ^long [uplo]
  (case uplo
    :upper 121
    :lower 122
    121 121
    122 122
    (throw (ex-info "Invalid uplo" {:uplo uplo}))))

(defn enc-diag ^long [diag]
  (case diag
    :unit 132
    :non-unit 131
    132 132
    131 131
    (throw (ex-info "Invalid diag" {:diag diag}))))
