(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.dummy-engine
  (:refer-clojure :exclude [accessor])
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.opencl.clblock :refer :all])
  (:import [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Block Matrix DataAccessor]))

(def ^String MSG_UNSUPPORTED "BLAS operations are not supported by dummy engines. Please use other engines if you need BLAS.")

(deftype DummyVectorEngine []

  Releaseable
  (release [_] true)
  BLAS
  (swap [_ _ y]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (copy [_ _ y]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (dot [_ _ y]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (nrm2 [_ _]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (asum [_ _]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (iamax [_ _]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (rot [_ _ y c s]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (rotg [_ _]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (rotm [_ _ y p]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (rotmg [_ _ args]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (scal [_ alpha _]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (axpy [_ alpha x y]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  BLASPlus
  (sum [_ x]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED))))

;; ======================= Dense Matrix ========================================

(deftype DummyMatrixEngine []
  Releaseable
  (release [_]
    true)
  BLAS
  (mv [_ _ _ _ _ _]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED)))
  (mm [_ _ _ _ _ _]
    (throw (UnsupportedOperationException. MSG_UNSUPPORTED))))

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
