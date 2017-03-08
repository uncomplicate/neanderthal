;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.lapack)

;; ================== Auxiliary LAPACK ==============================================

(defmacro ge-lan [method norm a]
  `(when (< 0 (.count ~a))
     (~method (.order ~a) ~norm (.mrows ~a) (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a))))

(defmacro ge-laset [method alpha beta a]
  `(when (< 0 (.count ~a))
     (~method (.order ~a) (int \g) (.mrows ~a) (.ncols ~a) ~alpha ~beta (.buffer ~a) (.offset ~a) (.stride ~a))))

;; ================== Singular Value Decomposition LAPACK ==============================================

(defmacro ge-sv
  ([method a b ipiv]
   `(~method (.order ~a) (.ncols ~a) (.ncols ~b) (.buffer ~a) (.offset ~a) (.stride ~a)
     (.buffer ~ipiv) (.offset ~ipiv) (.buffer ~b) (.offset ~b) (.stride ~b))))

(defmacro ge-trf
  ([method a ipiv]
   `(~method (.order ~a) (.mrows ~a) (.ncols ~a)
     (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~ipiv) (.offset ~ipiv))))
