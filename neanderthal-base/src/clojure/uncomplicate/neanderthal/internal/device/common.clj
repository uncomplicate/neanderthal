;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.device.common
  (:require [uncomplicate.neanderthal.internal.api
             :refer [compatible? fits? equals-block navigator region]])
  (:import [uncomplicate.neanderthal.internal.api
            MatrixImplementation LayoutNavigator Region Matrix]))

(defn name-transp
  ([condition name a b]
   (format "%s_%s" name (if condition "no_transp" "transp")))
  ([name a b]
   (name-transp (= (navigator a) (navigator b)) name a b)))

(defn layout-match? [a b]
  (= (navigator a) (navigator b)))

(defn symmetric-match? [a b]
  (if (= (navigator a) (navigator b))
    (= (.uplo (region a)) (.uplo (region b)))
    (not= (.uplo (region a)) (.uplo (region b)))))

(defn uplo-bottom? [a]
  (let [reg (region a)]
    (if (.isColumnMajor (navigator a)) (.isLower reg) (.isUpper reg))))

(defn device-matrix-equals [eng ^Matrix a ^Matrix b]
  (or (identical? a b)
      (and (instance? (class a) b)
           (= (.matrixType ^MatrixImplementation a) (.matrixType ^MatrixImplementation b))
           (compatible? a b) (= (.mrows a) (.mrows b)) (= (.ncols a) (.ncols b))
           (equals-block eng a b))))

(defn device-vector-equals [eng x y]
  (or (identical? x y)
      (and (instance? (class x) y) (compatible? x y) (fits? x y) (equals-block eng x y))))
