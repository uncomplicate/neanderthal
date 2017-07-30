;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.api
  (:require [uncomplicate.commons.core :refer [Releaseable]]))

(definterface UploNavigator ;;TODO obsolete see  Navigator
  (^boolean accessible [^long i ^long j])
  (^long colStart [^long n ^long i])
  (^long colEnd [^long n ^long i])
  (^long rowStart [^long n ^long i])
  (^long rowEnd [^long n ^long i])
  (^long defaultEntry [^long i ^long j])
  (^long diaDim [^long n ^long k]);;TODO it should be removed, since it is used at exactly one place (.dia)?
  (^long unitIndex [^long i]))

(definterface StripeNavigator;;TODO obsolete: see layoutnavigator
  (^long start [^long n ^long j])
  (^long end [^long n ^long j])
  (^long offsetPad [^long ld]))

(definterface RealOrderNavigator
  (^long sd [^long m ^long n]);TODO full storage
  (^long fd [^long m ^long n]);;TODO full storage
  (^long index [^long ofst ^long ld ^long k]);;TODO to storage
  (^long index [^long ofst ^long ld ^long i ^long j]);;TODO to storage
  (^double get [a ^long i ^long j]);;TODO real layout navigator
  (set [a ^long i ^long j ^double val]);;TODO real layout navigator
  (invokePrimitive [f ^long i ^long j ^double val]);;todo real layout navigator
  (stripe [a ^long j]));;todo layoutnavigator

(definterface BandNavigator;;TODO obsolete see navigators and storages
  (^long height [^long m ^long n ^long kl ^long ku]);;TODO banded stoarage
  (^long width [^long m ^long n ^long kl ^long ku]);;TODO band storage
  (^long sd [^long m ^long n ^long kl ^long ku]);;TODO full storage
  (^long fd [^long m ^long n ^long kl ^long ku]);;TODO full storage
  (^long index [^long ofst ^long ld ^long kl ^long ku ^long i ^long j]);;todo storage
  (^long index [^long ofst ^long ld ^long kl ^long ku ^long k]);;todo storage
  (^long stripeIndex [^long offset ^long ld ^long kl ^long ku ^long i ^long j]);;TODO unused!!!!!!
  (^long start [^long kl ^long ku ^long stripe]);;TODO layout navigator
  (^long end [^long m ^long n ^long kl ^long ku ^long stripe]);;TODO layout navigator
  (^long kd [^long kl ^long ku]));;TODO banded storage

;; ================================ Navigation ===================================

(definterface Region
  (^boolean accessible [^long i ^long j])
  (^long colStart [^long i])
  (^long colEnd [^long i])
  (^long rowStart [^long i])
  (^long rowEnd [^long i])
  (^boolean isUpper [])
  (^boolean isLower [])
  (^boolean isDiagUnit []))

(definterface LayoutNavigator
  (^long start [region ^long j])
  (^long end [region ^long j])
  (^long nstripes [a])
  (stripe [a ^long j])
  (^boolean isColumnMajor [])
  (^boolean isRowMajor []))

(definterface RealLayoutNavigator
  (^double get [a ^long i ^long j])
  (set [a ^long i ^long j ^double val])
  (invokePrimitive [f ^long i ^long j ^double val]))

;; ================================ Default entry =================================

(definterface RealDefault
  (entry [stor da buf ^long ofst ^long i ^long j]))

;; ================================ Storage =======================================

(definterface DenseStorage
  (^long fd [])
  (^long index [^long i ^long j])
  (^boolean isColumnMajor []) ;;TODO probably obsolete
  (^long layout [])) ;;TODO probably obsolete

;;TODO ld and stride... Block should be incorporated in this system

(definterface FullStorage
  (^long ld [])
  (^long sd []))

(definterface BandStorage
  (^long height [])
  (^long width [])
  (^long kd []))

(defprotocol StorageProvider;;TODO rename
  (region ^Region [this])
  (storage ^DenseStorage [this])
  (navigator ^LayoutNavigator [this]))


(defprotocol Navigable ;;TODO probably remove after refactoring to navigators and storages
  (order-navigator ^RealOrderNavigator [this])
  (stripe-navigator ^StripeNavigator [this])
  (uplo-navigator ^UploNavigator [this])
  (band-navigator ^BandNavigator [this]))

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
  (rk [this alpha x y a])
  (mm [this alpha a b beta c left] [this alpha a b left]))

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
  (trf [this a ipiv])
  (tri [this a ipiv] [this a])
  (det [this a ipiv] [this a])
  (trs [this a b ipiv] [this a b])
  (con [this lu ipiv nrm nrm1?][this lu nrm nrm1?] [this a nrm1?])
  (sv [this a b pure])
  (qrf [this a tau])
  (qrfp [this a tau])
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
  (ev [this a w vl vr])
  (svd [this a s superb] [this a s u vt superb]))

(defprotocol TRF
  (trtrs [a b])
  (trtri! [a])
  (trtri [a])
  (trcon [a nrm nrm1?] [a nrm1?])
  (trdet [a]))

(defprotocol BlockEngine
  (equals-block [_ cu-x cu-y]))

(defprotocol ReductionFunction
  (vector-reduce [f init x] [f init x y] [f init x y z] [f init x y z v])
  (vector-reduce-map [f init g x] [f init g x y] [f init g x y z] [f init g x y z v]))

(defprotocol Factory
  (create-vector [this n init])
  (create-ge [this m n ord init])
  (create-tr [this n ord uplo diag init])
  (create-sy [this n ord uplo init])
  (create-gb [this m n kl ku ord init])
  (create-tb [this n k ord uplo init])
  (create-sb [this n k ord uplo init])
  (create-packed [this n matrix-type column? lower? init])
  (vector-engine [this])
  (ge-engine [this])
  (tr-engine [this])
  (sy-engine [this])
  (gb-engine [this])
  (tb-engine [this])
  (sb-engine [this])
  (tp-engine [this])
  (sp-engine [this]))

(defprotocol EngineProvider
  (engine [this]))

(defprotocol FactoryProvider
  (factory [this])
  (native-factory [this])
  (index-factory [this]))

(defprotocol DataAccessorProvider
  (data-accessor ^DataAccessor [this]))

(defprotocol Info
  (info [this]))

(defprotocol MemoryContext
  (compatible? [this other])
  (fits? [this other])
  (fits-navigation? [this other]);; TODO remove. obsolete....
  (fully-packed? [this]));; TODO move to storage/navigation/whatever

(defprotocol Container
  (raw [this] [this factory])
  (zero [this] [this factory])
  (host [this])
  (native [this]))

(defprotocol DenseContainer
  (view-ge [this] [this stride-mult])
  (view-tr [this uplo diag])
  (view-sy [this uplo])
  (view-vctr [this] [this stride-mult])
  (view-gb [this kl ku] [this]))

;; ============ Realeaseable ===================================================

(extend-type clojure.lang.Sequential
  Releaseable
  (release [this]
    true)
  Container
  (raw [this fact]
    (let [n (count this)]
      (create-vector fact n false))))

(extend-type Object
  MemoryContext
  (compatible? [this o]
    (instance? (class this) o))
  DataAccessorProvider
  (data-accessor [_]
    nil))

(extend-type nil
  MemoryContext
  (compatible? [this o]
    false)
  DataAccessorProvider
  (data-accessor [_]
    nil))

;; ============================================================================

(def ^:const ROW_MAJOR 101)
(def ^:const COLUMN_MAJOR 102)
(def ^:const DEFAULT_ORDER COLUMN_MAJOR)

(def ^:const UPPER 121)
(def ^:const LOWER 122)
(def ^:const DEFAULT_UPLO LOWER)

(def ^:const DIAG_NON_UNIT 131)
(def ^:const DIAG_UNIT 132)
(def ^:const DEFAULT_DIAG DIAG_NON_UNIT)

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

(defn enc-order ^long [order]
  (case order
    :row 101
    :column 102
    101 101
    102 102
    (throw (ex-info "Invalid order" {:order order}))))

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

(defn flip ^long [^long property]
  (case property
    101 102
    102 101
    111 112
    112 111
    113 111
    121 122
    122 121
    131 132
    132 131
    (throw (ex-info "Invalid property" {:property property}))))

(defn flip-layout ^long [^long layout] ;;TODO obsolete
  (case layout
    101 102
    102 101
    (throw (ex-info "Invalid layout" {:layout layout}))))

(defn flip-uplo ^long [^long uplo];;TODO obsolete
  (case uplo
    121 122
    122 121
    (throw (ex-info "Invalid uplo" {:uplo uplo}))))
