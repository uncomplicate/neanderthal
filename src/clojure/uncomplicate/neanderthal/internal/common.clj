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
   (dragan-says-ex message nil)))

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

(defrecord TRFactorization [^Matrix lu ^Vector ipiv ^Boolean master fresh]
  Releaseable
  (release [_]
    (when master (release lu))
    (release ipiv))
  TRF
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
      (con (engine lu) lu nrm nrm1?)
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

(defn create-trf
  ([lu a ipiv]
   (->TRFactorization lu ipiv true (atom true)))
  ([lu ipiv]
   (->TRFactorization lu ipiv false (atom true))))
