(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.dummy-engine
  (:refer-clojure :exclude [accessor])
  (:require [uncomplicate.clojurecl.core :refer [Releaseable release]]
            [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.opencl.clblock :refer :all])
  (:import [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Block Matrix DataAccessor]))

(def ^String UNSUPPORTED_MSG "BLAS operations are not supported by dummy-engine.
Please use other OpenCL engines if you need BLAS operations.")

(deftype DummyVectorEngine []
  Releaseable
  (release [_] true)
  BLAS
  (swap [_ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (copy [_ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (dot [_ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (nrm2 [_ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (asum [_ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (iamax [_ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (rot [_ _ _ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (rotg [_ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (rotm [_ _ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (rotmg [_ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (scal [_ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (axpy [_ _ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  BLASPlus
  (subcopy [_ _ _ _ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (sum [_ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (imax [_ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (imin [_ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG))))

;; ======================= Dense Matrix ========================================

(deftype DummyMatrixEngine []
  Releaseable
  (release [_]
    true)
  BLAS
  (mv [_ _ _ _ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG)))
  (mm [_ _ _ _ _ _]
    (throw (UnsupportedOperationException. UNSUPPORTED_MSG))))

(def ^:private dummy-vector-engine (->DummyVectorEngine))
(def ^:private dummy-matrix-engine (->DummyMatrixEngine))

(deftype DummyFactory [claccessor ctx queue]
  Releaseable
  (release [_] true)
  Factory
  (data-accessor [_]
    claccessor)
  (vector-engine [_ _]
    dummy-vector-engine)
  (matrix-engine [_ _]
    dummy-matrix-engine))

(defn dummy-factory
  ([create-accessor ctx queue]
   (->DummyFactory (create-accessor ctx queue) ctx queue)))
