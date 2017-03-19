;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.linalg
  "Contains type-agnostic linear algebraic functions roughly corresponding to the functionality
  usually defined in LAPACK (factorizations, solvers, etc.). This namespace works similarily
  to the core namespace; see there for more details about the intended use.
  "
  (:require [uncomplicate.commons.core :refer [let-release]]
            [uncomplicate.neanderthal.core :refer [vctr ge]]
            [uncomplicate.neanderthal.internal.api :as api])
  (:import [uncomplicate.neanderthal.internal.api Vector Matrix GEMatrix TRMatrix Changeable]))

;; ============================= LAPACK =======================================

;; ------------- Singular Value Decomposition LAPACK -------------------------------

(defn trf!
  "TODO"
  (^Vector [^Matrix a ^Vector ipiv]
   (if (= (.ncols a) (.dim ipiv))
     (api/trf (api/engine a) a ipiv)
     (throw (IllegalArgumentException. "TODO"))))
  (^Vector [^Matrix a]
   (let-release [ipiv (vctr (api/index-factory a) (.ncols a))]
     (trf! a ipiv))))

(defn trs!
  "TODO"
  (^Vector [^Matrix a ^Matrix b ^Vector ipiv]
   (if (and (= (.ncols a) (.mrows b) (.dim ipiv)) (api/fits-navigation? a b))
     (api/trs (api/engine a) a b ipiv)
     (throw (IllegalArgumentException. "TODO"))))
  (^Vector [^Matrix a b]
   (let-release [ipiv (vctr (api/index-factory a) (.ncols a))]
     (trs! a b ipiv))))

(defn sv!
  "TODO"
  (^Vector [^Matrix a ^Matrix b ^Vector ipiv]
   (if (and (= (.ncols a) (.mrows b) (.dim ipiv)) (api/fits-navigation? a b))
     (api/sv (api/engine a) a b ipiv)
     (throw (IllegalArgumentException. "TODO"))))
  (^Vector [^Matrix a b]
   (let-release [ipiv (vctr (api/index-factory a) (.ncols a))]
     (sv! a b ipiv))))

;; ------------- Orthogonal Factorization (L, Q, R) LAPACK -------------------------------

(defn ^:private min-mn ^long [^Matrix a]
  (max 1 (min (.mrows a) (.ncols a))))

(defn qrf!
  "TODO"
  ([^Matrix a ^Vector tau]
   (if (and (= (.dim tau) (min-mn a)))
     (api/qrf (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (qrf! a tau))))

(defn qrfp!
  "TODO"
  ([^Matrix a ^Vector tau]
   (if (and (= (.dim tau) (min-mn a)))
     (api/qrfp (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (qrfp! a tau))))

(defn rqf!
  "TODO"
  ([^Matrix a ^Vector tau]
   (if (and (= (.dim tau) (min-mn a)))
     (api/rqf (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (rqf! a tau))))

(defn lqf!
  "TODO"
  ([^Matrix a ^Vector tau]
   (if (and (= (.dim tau) (min-mn a)))
     (api/lqf (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (lqf! a tau))))

(defn qlf!
  "TODO"
  ([^Matrix a ^Vector tau]
   (if (and (= (.dim tau) (min-mn a)))
     (api/qlf (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (qlf! a tau))))

(defn ls!
  "TODO"
  [^Matrix a ^Matrix b]
  (if (and (<= (max 1 (.mrows a) (.ncols a)) (.mrows b)) (api/fits-navigation? a b))
    (api/ls (api/engine a) a b)
    (throw (IllegalArgumentException. "TODO"))))

(defn ev!
  "TODO"
  ([^Matrix a ^Matrix w ^Matrix vl ^Matrix vr]
   (if (and (= (.mrows a) (.ncols a))
            (= (.mrows a) (.mrows w)) (= 2 (.ncols w))
            (or (nil? vl) (and (= (.mrows a) (.mrows vl) (.ncols vl)) (api/fits-navigation? a vl)))
            (or (nil? vr) (and (= (.mrows a) (.mrows vr) (.ncols vr)) (api/fits-navigation? a vr))))
     (api/ev (api/engine a) a w vl vr)
     (throw (IllegalArgumentException. "TODO not square matrix a."))))
  ([a w]
   (ev! a w nil nil))
  ([^Matrix a vl vr]
   (let-release [w (ge (api/factory a) (.mrows a) 2)]
     (ev! a w vl vr)))
  ([^Matrix a]
   (ev! a nil nil)))

(defn svd!
  "TODO"
  ([^Matrix a ^Vector s ^Matrix u ^Matrix vt ^Vector superb]
   (let [m (.mrows a)
         n (.ncols a)
         min-mn (min m n)]
     (if (and (or (nil? u) (and (or (= m (.mrows u) (.ncols u))
                                    (and (= m (.mrows u)) (= min-mn (.ncols u))))
                                (api/fits-navigation? a u)))
              (or (nil? vt) (and (or (= n (.mrows vt) (.ncols vt))
                                     (and (= min-mn (.mrows vt)) (= n (.ncols vt))))
                                 (api/fits-navigation? a vt)))
              (= min-mn (.dim s) (.dim superb)))
       (api/svd (api/engine a) a s u vt superb)
       (throw (IllegalArgumentException. "TODO detailed error.")))))
  ([^Matrix a ^Vector s ^Vector superb]
   (if (and (= (min (.mrows a) (.ncols a)) (.dim s) (.dim superb)))
       (api/svd (api/engine a) a s superb)
       (throw (IllegalArgumentException. "TODO detailed error."))))) ;;TODO create other arities
