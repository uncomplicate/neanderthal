;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.api
  (:require [uncomplicate.commons.core :refer [Releaseable]]))

(definterface UploNavigator
  (^long colStart [^long n ^long i])
  (^long colEnd [^long n ^long i])
  (^long rowStart [^long n ^long i])
  (^long rowEnd [^long n ^long i])
  (^long defaultEntry [^long i ^long j])
  (^long unitIndex [^long i]))

(definterface StripeNavigator
  (^long start [^long n ^long j])
  (^long end [^long n ^long j]))

(definterface RealOrderNavigator
  (^long sd [^long m ^long n])
  (^long fd [^long m ^long n])
  (^long index [^long ofst ^long ld ^long i ^long j])
  (^double get [a ^long i ^long j])
  (set [a ^long i ^long j ^double val])
  (invokePrimitive [f ^long i ^long j ^double val])
  (stripe [a ^long j]))

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
  (mm [this alpha a b beta c] [this alpha a b left]))

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
  (inv [this a])
  (trs [this a b ipiv] [this a b])
  (con [this lu nrm nrm1?] [this a nrm1?])
  (sv [this a b ipiv] [this a b])
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

(defprotocol BlockEngine
  (equals-block [_ cu-x cu-y]))

(defprotocol ReductionFunction
  (vector-reduce [f init x] [f init x y] [f init x y z] [f init x y z v])
  (vector-reduce-map [f init g x] [f init g x y] [f init g x y z] [f init g x y z v]))

(defprotocol Factory
  (create-vector [this n init])
  (create-ge [this m n ord init])
  (create-tr [this n ord uplo diag init])
  (vector-engine [this])
  (ge-engine [this])
  (tr-engine [this]))

(defprotocol EngineProvider
  (engine [this]))

(defprotocol FactoryProvider
  (factory [this])
  (native-factory [this])
  (index-factory [this]))

(defprotocol DataAccessorProvider
  (data-accessor ^DataAccessor [this]))

(defprotocol MemoryContext
  (compatible? [this other])
  (fits? [this other])
  (fits-navigation? [this other])
  (fully-packed? [this]))

(defprotocol Container
  (raw [this] [this factory])
  (zero [this] [this factory])
  (host [this])
  (native [this]))

(defprotocol DenseContainer
  (view-ge [this] [this stride-mult])
  (view-tr [this uplo diag])
  (view-vctr [this] [this stride-mult]))

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
