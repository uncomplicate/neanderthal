;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.common
  (:require [uncomplicate.fluokitten.core :refer [fold]]
            [uncomplicate.commons.core :refer [Releaseable release let-release double-fn]]
            [uncomplicate.neanderthal.internal.api :refer :all])
  (:import [uncomplicate.neanderthal.internal.api Matrix Vector Region RealBufferAccessor]))

(defn dragan-says-ex
  ([message data]
   (throw (ex-info (format "Dragan says: %s" message) data)))
  ([message]
   (dragan-says-ex message {})))

;; ================= Core Functions ===================================

(defn dense-rows [^Matrix a]
  (map #(.row a %) (range (.mrows a))))

(defn dense-cols [^Matrix a]
  (map #(.col a %) (range (.ncols a))))

(defn dense-dias [^Matrix a]
  (map #(.dia a %) (range (dec (.ncols a)) (- (.mrows a)) -1)))

(defn region-rows [^Matrix a]
  (map #(.row a %) (range (min (.mrows a) (+ (min (.mrows a) (.ncols a)) (.kl (region a)))))))

(defn region-cols [^Matrix a]
  (map #(.col a %) (range (min (.ncols a) (+ (min (.mrows a) (.ncols a)) (.ku (region a)))))))

(defn region-dias [^Matrix a]
  (let [reg (region a)]
    (map #(.dia a %) (range (.ku reg) (- (inc (.kl reg))) -1))))

(defn ^RealBufferAccessor real-accessor [a]
  (data-accessor a))

;; ======================== LU factorization ==========================================

(def ^:private f* (double-fn *))
(def ^:private falsify (constantly false))

(defn ^:private stale-factorization []
  (throw (ex-info "Cannot compute with stale LU factorization." {})))

(defn ^:private nrm-needed-for-con []
  (throw (ex-info "Cannot compute condition number without nrm." {})))

(defrecord LUFactorization [^Matrix lu ^Vector ipiv ^Boolean master fresh]
  Releaseable
  (release [_]
    (when master (release lu))
    (release ipiv))
  Info
  (info [this]
    this)
  TRF
  (create-trf [this _ _]
    this)
  (trtrs [_ b]
    (if @fresh
      (trs (engine lu) lu b ipiv)
      (stale-factorization)))
  (trtri! [_]
    (if (compare-and-set! fresh true false)
      (tri (engine lu) lu ipiv)
      (stale-factorization)))
  (trtri [_]
    (if @fresh
      (let-release [res (raw lu)]
        (let [eng (engine lu)]
          (tri eng (copy eng lu res) ipiv))
        res)
      (stale-factorization)))
  (trcon [_ nrm nrm1?]
    (if @fresh
      (con (engine lu) lu ipiv nrm nrm1?)
      (stale-factorization)))
  (trcon [_ nrm1?]
    (nrm-needed-for-con))
  (trdet [_]
    (if @fresh
      (let [res (double (fold f* 1.0 (.dia lu)))]
        (if (even? (.dim ipiv))
          res
          (- res)))
      (stale-factorization)))
  Matrix
  (mrows [_]
    (.mrows lu))
  (ncols [_]
    (.ncols lu))
  MemoryContext
  (compatible? [_ b]
    (compatible? lu b))
  (fits? [_ b]
    (fits? lu b))
  (fits-navigation? [_ b]
    (fits-navigation? lu b)))

(defrecord CholeskyFactorization [^Matrix gg ^Boolean master fresh]
  Releaseable
  (release [_]
    (if master (release gg) true))
  Info
  (info [this]
    this)
  POTRF
  (create-cholesky [this _ _]
    this)
  TRF
  (create-trf [this _ _]
    this)
  (trtrs [_ b]
    (if @fresh
      (trs (engine gg) gg b)
      (stale-factorization)))
  (trtri! [_]
    (if (compare-and-set! fresh true false)
      (tri (engine gg) gg)
      (stale-factorization)))
  (trtri [_]
    (if @fresh
      (let-release [res (raw gg)]
        (let [eng (engine gg)]
          (tri eng (copy eng gg res)))
        res)
      (stale-factorization)))
  (trcon [_ nrm nrm1?]
    (if @fresh
      (con (engine gg) gg nrm nrm1?)
      (stale-factorization)))
  (trcon [_ nrm1?]
    (nrm-needed-for-con))
  (trdet [_]
    (if @fresh
      (let [dia-gg (.dia gg)
            res (double (fold f* 1.0 dia-gg))]
        (if (even? (.dim dia-gg))
          res
          (- res)))
      (stale-factorization)))
  Matrix
  (mrows [_]
    (.mrows gg))
  (ncols [_]
    (.ncols gg))
  MemoryContext
  (compatible? [_ b]
    (compatible? gg b))
  (fits? [_ b]
    (fits? gg b))
  (fits-navigation? [_ b]
    (fits-navigation? gg b)))
