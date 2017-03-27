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
  usually defined in LAPACK (factorizations, solvers, etc.). This namespace works similarly
  to the [[uncomplicate.neanderthal.core]] namespace; see there for more details about the intended use.

  ### Cheat Sheet

  - Linear equations and LU factorization: [[trf!]], [[trs!]], [[sv!]].
  - Orthogonal factorizations: [[qrf!]], [[qrfp!]], [[gqr!]], [[mqr!]], [[rqf!]], [[grq!]], [[mrq!]],
   [[qlf!]], [[gql!]], [[mql!]], [[lqf!]], [[glq!]], [[mlq!]].
  - Linear solver: [[ls!]].
  - Eigen decomposition: [[ev!]].
  - Singular value decomposition (SVD): [[svd!]].

  ### Also see:

  - [LAPACK routines](https://software.intel.com/en-us/node/520866)
  - [Linear Equation Computational Routines](https://software.intel.com/en-us/node/520875)
  - [Linear Equations](https://software.intel.com/en-us/node/520972)
  - [Orthogonal Factorizations (Q, R, L)](https://software.intel.com/en-us/node/521003)
  - [Singular Value Decomposition](https://software.intel.com/en-us/node/521036)
  - [Symmetric Eigenvalue Problems](https://software.intel.com/en-us/node/521119)
  - Other LAPACK documentation, as needed.
  "
  (:require [uncomplicate.commons
             [core :refer [let-release]]
             [utils :refer [cond-into]]]
            [uncomplicate.neanderthal.core :refer [vctr vctr? ge]]
            [uncomplicate.neanderthal.internal.api :as api])
  (:import [uncomplicate.neanderthal.internal.api Vector Matrix GEMatrix TRMatrix Changeable]))

;; ============================= LAPACK =======================================

;; ------------- Singular Value Decomposition LAPACK -------------------------------

(defn trf!
  "Computes the LU factorization of a GE `mxn` matrix `a`.

  Overwrites `a` with L and U. L is stored as a lower unit triangle, and U as an upper triangle.
  Pivot is  placed into the `ipiv`, a vector of **integers or longs**.

  If the stride of `ipiv` is not `1`, or its dimension is not equals to `a`'s number of columns,
  or some value is illegal, throws ExceptionInfo.
  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [lapacke_?getrf](https://software.intel.com/en-us/node/520877).
  "
  (^Vector [^Matrix a ^Vector ipiv]
   (if (= (.ncols a) (.dim ipiv))
     (api/trf (api/engine a) a ipiv)
     (throw (ex-info "Column number of a and the dimension of ipiv do not fit."
                     {:n (.ncols a) :dim (.dim ipiv)}))))
  (^Vector [^Matrix a]
   (let-release [ipiv (vctr (api/index-factory a) (.ncols a))]
     (trf! a ipiv))))

(defn trs!
  "Solves a system of linear equations with an LU factored matrix `lu`, with multiple right
  hand sides matrix `b`. Overwrites `b` by the solution matrix.

  If the stride of `ipiv` is not `1`, or its dimension is not equeals to `lu`'s number of columns,
  or some value is illegal, throws ExceptionInfo.
  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [lapacke_?getrs](https://software.intel.com/en-us/node/520892).
  "
  (^Vector [^Matrix lu ^Matrix b ^Vector ipiv]
   (if (and (= (.ncols lu) (.mrows b) (.dim ipiv)) (api/fits-navigation? lu b))
     (api/trs (api/engine lu) lu b ipiv)
     (throw (ex-info "Column number of a and the dimension of ipiv do not fit."
                     {:n (.ncols lu) :dim (.dim ipiv)}))))
  (^Vector [^Matrix lu b]
   (let-release [ipiv (vctr (api/index-factory lu) (.ncols lu))]
     (trs! lu b ipiv))))

(defn sv!
  "Solves a system of linear equations with a square coefficient matrix `a` and multiple right
  hand sides matrix `b`. Overwrites `b` by the solution matrix.

  Overwrites `a` with L and U, `b` with the solution, and `ipiv` with a pivoting vector.
  L is stored as a lower unit triangle, and U as an upper triangle. Pivot is  placed into
  the `ipiv`, a vector of **integers or longs**.

  If the stride of `ipiv` is not `1`, or its dimension is not equeals to `a`'s number of columns,
  or some value is illegal, throws ExceptionInfo.
  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [lapacke_?gesv](https://software.intel.com/en-us/node/520973).
  "
  (^Vector [^Matrix a ^Matrix b ^Vector ipiv]
   (if (and (= (.ncols a) (.mrows b) (.dim ipiv)) (api/fits-navigation? a b))
     (api/sv (api/engine a) a b ipiv)
     (throw (ex-info "Column number of a and the dimension of ipiv do not fit."
                     {:n (.ncols a) :dim (.dim ipiv)}))))
  (^Vector [^Matrix a b]
   (let-release [ipiv (vctr (api/index-factory a) (.ncols a))]
     (sv! a b ipiv))))

;; ------------- Orthogonal Factorization (L, Q, R) LAPACK -------------------------------

(defn ^:private min-mn ^long [^Matrix a]
  (max 1 (min (.mrows a) (.ncols a))))

(defmacro ^:private with-lqrf-check [a tau expr]
  `(if (and (= (.dim ~tau) (min-mn ~a)))
     ~expr
     (throw (ex-info "Dimension of tau is not equals to (min ncols mrows) of a."
                     {:m (.mrows ~a) :n (.ncols ~a) :dim (.dim ~tau)}))))

(defmacro ^:private with-gq*-check [a tau expr]
  `(if (<= 1 (.dim ~tau) (.ncols ~a) (.mrows ~a))
     ~expr
     (throw (ex-info "Dimensions of a and tau do not fit requirements for computing Q."
                     {:m (.mrows ~a) :n (.ncols ~a) :dim (.dim ~tau) :errors
                      (cond-into []
                                 (not (<= (.dim ~tau) (.ncols ~a))) "(not (<= (dim tau) (ncols a)))"
                                 (not (<= (.ncols ~a) (.mrows ~a))) "(not (<= (ncols ~a) (mrows a)))")}))))

(defmacro ^:private with-g*q-check [a tau expr]
  `(if (<= (.dim ~tau) (.mrows ~a) (.ncols ~a))
     ~expr
     (throw (ex-info "Dimensions of a and tau do not fit requirements for computing Q."
                     {:m (.mrows ~a) :n (.ncols ~a) :dim (.dim ~tau) :errors
                      (cond-into []
                                 (not (<= (.dim ~tau) (.mrows ~a))) "(not (<= (dim tau) (mrows a)))"
                                 (not (<= (.mrows ~a) (.ncols ~a))) "(not (<= (mrows a) (ncols a)))")}))))

(defmacro ^:private with-mq*-check [a tau c left expr]
  `(let [r# (long (if ~left (.mrows ~c) (.ncols ~c)))]
     (if (and (<= (.dim ~tau) r#) (= r# (.mrows ~a)) (= (.ncols ~a) (.dim ~tau)))
       ~expr
       (throw (ex-info "Dimensions of a, tau, and c do not fit matrix multiplication."
                       {:a (str ~a) :c (str ~c) :tau (str ~tau) :errors
                        (cond-into []
                                   (not (<= (.dim ~tau) r#))
                                   (format "(not (<= (dim tau) %s))" (if ~left "(mrows c)" "(ncols c)"))
                                   (not (= r# (.mrows ~a)))
                                   (format "(not (= %s (mrows a)))" (if ~left "(mrows c)" "(ncols c)"))
                                   (not (= (.ncols ~a) (.dim ~tau))) "(not (= (ncols a) (dim tau)))")})))))

(defmacro ^:private with-m*q-check [a tau c left expr]
  `(let [r# (long (if ~left (.mrows ~c) (.ncols ~c)))]
     (if (and (<= (.dim ~tau) r#) (= (.mrows ~c) (.ncols ~a)) (= (.mrows ~a) (.dim ~tau)))
       ~expr
       (throw (ex-info "Dimensions of a, tau, and c do not fit matrix multiplication."
                       {:a (str ~a) :c (str ~c) :tau (str ~tau) :errors
                        (cond-into []
                                   (not (<= (.dim ~tau) r#))
                                   (format "(not (<= (dim tau) %s))" (if ~left "(mrows c)" "(ncols c)"))
                                   (not (= (.dim ~tau) (.mrows ~a))) "(not (= (dim tau) (mrows a)))"
                                   (not (= (.mrows ~c) (.ncols ~a))) "(not (= (mrows c) (ncols a)))")})))))

(defn qrf!
  "Computes the QR factorization of a GE `m x n` matrix.

  The input is a GE matrix `a`. The output overwrites the contents of `a`. Output QR is laid out
  in `a` in the following way: The elements in the upper triangle (or trapezoid) contain the
  `(min m n) x n` upper triangular (or trapezoidal) matrix R. The elements in the lower triangle
  (or trapezoid) **below the diagonal**, with the vector `tau` contain Q as a product of `(min m n)`
  elementary reflectors. **Other routines work with Q in this representation**. If you need
  to compute q, call [[gqr!]].

  If the stride of `tau` is not `1`, or some value is illegal, throws ExceptionInfo.

  See related info about [lapacke_?geqrf](https://software.intel.com/en-us/node/521004).
  "
  ([^Matrix a ^Vector tau]
   (with-lqrf-check a tau (api/qrf (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (qrf! a tau))))

(defn qrfp!
  "Computes the QR factorization of a GE `m x n` matrix, with non-negative diagonal elements.

  See [[qrf!]].

  See related info about [lapacke_?geqrfp](https://software.intel.com/en-us/node/468946).
  "
  ([^Matrix a ^Vector tau]
   (with-lqrf-check a tau (api/qrfp (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (qrfp! a tau))))

(defn gqr!
  "Generates the real orthogonal matrix Q of the QR factorization formed by [[qrf!]] or [[qrfp!]].

  The input is a GE matrix `a` and vector `tau` that were previously processed by [[qrf!]]. The matrix
  Q is square, `m x m` dimensional, but if the dimension of factorization is `m x n`, only the leading
  `n` columns of Q are computed. The output overwrites `a` with the leading `n` columns of Q which
  form an orthonormal basis in the space spanned by columns of `a`.

  If the condition `(<= (dim tau) (ncols a) (mrows a))` is not satisfied, throws ExceptionInfo.''
  If the stride of `tau` is not `1`, or some value is illegal, throws ExceptionInfo.

  See related info about [lapacke_?orgqr](https://software.intel.com/en-us/node/468956).
  "
  [^Matrix a ^Vector tau]
  (with-gq*-check a tau (api/gqr (api/engine a) a tau)))

(defn mqr!
  "Multiplies a real matrix by the orthogonal matrix Q formed by [[qrf!]] or [[qrfp!]].

  The input is a GE matrix `a` and vector `tau` that were previously processed by [[qrf!]],
  and a matrix C. The side of the multiplication (CQ or QC) is determined by the position of the vector
  (also the output of [[qrf!]]), which, by convenience, is after Q: the input `x y z` is either
  C Q tau, or Q tau C. The output overwrites C with the product.

  The dimensions of C, Q, and tau must satisfy the following conditions, or ExceptionInfo is thrown:

  * `(= (ncols a) (dim tau))`
  * for Q tau C (multiplication from the left):
      - `(<= (dim tau) (mrows c))`
      - `(= (mrows C) (mrows a))`
  * for C Q tau (multiplication from the right):
       `(<= (dim tau) (ncols c))`
      - `(= (ncols C) (mrows a))`

  If the stride of `tau` is not `1`, or some value is illegal, throws ExceptionInfo.

  See related info about [lapacke_?ormqr](https://software.intel.com/en-us/node/468958).
  "
  [x y z]
  (if (vctr? y)
    (with-mq*-check ^Matrix x ^Vector y ^Matrix z true (api/mqr (api/engine x) x y z true))
    (with-mq*-check ^Matrix y ^Vector z ^Matrix x false (api/mqr (api/engine y) y z x false))))

(defn rqf!
  "Computes the RQ factorization of a GE `m x n` matrix.

  See [[qrf!]].

  See related info about [lapacke_?gerqf](https://software.intel.com/en-us/node/521024).
  "
  ([^Matrix a ^Vector tau]
   (with-lqrf-check a tau (api/rqf (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (rqf! a tau))))

(defn grq!
  "Generates the real orthogonal matrix Q of the RQ factorization formed by [[rqf!]].

  The input is a GE matrix `a` and vector `tau` that were previously processed by [[rqf!]]. The matrix
  Q is square, `m x m` dimensional, but if the dimension of factorization is `m x n`, only the last
  `m` rows of Q are computed. The output overwrites `a` with the last `m` rows of Q which
  form an orthonormal basis in the space spanned by rows of `a`.

  If the condition `(<= (dim tau) (mrows a) (ncols a))` is not satisfied, throws ExceptionInfo.''
  If the stride of `tau` is not `1`, or some value is illegal, throws ExceptionInfo.

  See related info about [lapacke_?orgrq](https://software.intel.com/en-us/node/468986).
  "
  [^Matrix a ^Vector tau]
  (with-g*q-check a tau (api/grq (api/engine a) a tau)))

(defn mrq!
  "Multiplies a real matrix by the orthogonal matrix Q formed by [[rqf!]].

  The input is a GE matrix `a` and vector `tau` that were previously processed by [[rqf!]],
  and a matrix C. The side of the multiplication (CQ or QC) is determined by the position of the vector
  (also the output of [[rqf!]]), which, by convenience, is after Q: the input `x y z` is either
  C Q tau, or Q tau C. The output overwrites C with the product.

  The dimensions of C, Q, and tau must satisfy the following conditions, or ExceptionInfo is thrown:
  * `(= (mrows C) (ncols a))`
  * `(= (mrows a) (dim tau))`
  * for Q tau C (multiplication from the left):
      - `(<= (dim tau) (mrows c))`
  * for C Q tau (multiplication from the right):
      - `(<= (dim tau) (ncols c))`

  If the stride of `tau` is not `1`, or some value is illegal, throws ExceptionInfo.

  See related info about [lapacke_?ormrq](https://software.intel.com/en-us/node/468990).
  "
  [x y z]
  (if (vctr? y)
    (with-m*q-check ^Matrix x ^Vector y ^Matrix z true (api/mrq (api/engine x) x y z true))
    (with-m*q-check ^Matrix y ^Vector z ^Matrix x false (api/mrq (api/engine y) y z x false))))

(defn qlf!
  "Computes the QL factorization of a GE `m x n` matrix.

  See [[qrf!]].

  See related info about [lapacke_?geqlf](https://software.intel.com/en-us/node/521019).
  "
  ([^Matrix a ^Vector tau]
   (with-lqrf-check a tau (api/qlf (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (qlf! a tau))))

(defn gql!
  "Generates the real orthogonal matrix Q of the QL factorization formed by [[qlf!]].

  See [[gqr!]].

  See related info about [lapacke_?orgql](https://software.intel.com/en-us/node/468976).
  "
  [^Matrix a ^Vector tau]
  (with-gq*-check a tau (api/gql (api/engine a) a tau)))

(defn mql!
  "Multiplies a real matrix by the orthogonal matrix Q formed by [[qlf!]].

  See [[mqr!]].

  See related info about [lapacke_?ormql](https://software.intel.com/en-us/node/468980).
  "
  [x y z]
  (if (vctr? y)
    (with-mq*-check ^Matrix x ^Vector y ^Matrix z true (api/mql (api/engine x) x y z true))
    (with-mq*-check ^Matrix y ^Vector z ^Matrix x false (api/mql (api/engine y) y z x false))))

(defn lqf!
  "Computes the LQ factorization of a GE `m x n` matrix.

  See [[qrf!]].

  See related info about [lapacke_?gelqf](https://software.intel.com/en-us/node/521014).
  "
  ([^Matrix a ^Vector tau]
   (with-lqrf-check a tau (api/lqf (api/engine a) a tau)))
  ([a]
   (let-release [tau (vctr (api/factory a) (min-mn a))]
     (lqf! a tau))))

(defn glq!
  "Generates the real orthogonal matrix Q of the LQ factorization formed by [[lqf!]].

  See [[grq!]].

  See related info about [lapacke_?orglq](https://software.intel.com/en-us/node/468966).
  "
  [^Matrix a ^Vector tau]
  (with-g*q-check a tau (api/glq (api/engine a) a tau)))

(defn mlq!
  "Multiplies a real matrix by the orthogonal matrix Q formed by [[lqf!]].

  See [[mrq!]].

  See related info about [lapacke_?ormlq](https://software.intel.com/en-us/node/468968).
  "
  [^Matrix x y z]
  (if (vctr? y)
    (with-m*q-check ^Matrix x ^Vector y ^Matrix z true (api/mlq (api/engine x) x y z true))
    (with-m*q-check ^Matrix y ^Vector z ^Matrix x false (api/mlq (api/engine y) y z x false))))

(defn ls!
  "Solves an overdetermined or underdetermined linear system `AX = B` with full rank matrix using
  QR or LQ factorization.

  Overwrites `a` with the factorization data:
  - QR if `m >= n`;
  - LQ if `m < n`.

  Overwrites b with:
  - the least squares solution vectors if `m >= n`
  - minimum norm solution vectors if `m < n`.

  If `a` and `b` do not have the same order (column or row oriented), throws ExceptionInfo.
  If the `i`-th element of the triangular factor of a is zero, so that `a` does not have full rank,
  the least squares cannot be computed, and the function throws ExceptionInfo.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info about [lapacke_?gels](https://software.intel.com/en-us/node/521112).
  "
  [^Matrix a ^Matrix b]
  (if (and (<= (max 1 (.mrows a) (.ncols a)) (.mrows b)) (api/fits-navigation? a b))
    (api/ls (api/engine a) a b)
    (throw (ex-info "You cannot solve linear system described by incompatible or ill-fitting matrices."
                    {:a (str a) :b (str b) :errors
                     (cond-into []
                                (not (<= (max 1 (.mrows a) (.ncols a)) (.mrows b)))
                                "dimensions of a and b do not fit"
                                (not (api/fits-navigation? a b)) "a and b do not have the same orientation")}))))

(defn ev!
  "Computes the eigenvalues and left and right eigenvectors of a matrix `a`.

  On exit, `a` is overwritten with QR factors. The first 2 columns of a column-oriented GE matrix
  `w` are overwritten with eigenvalues of `a`. If `vl` and `vr` GE matrices are provided, they will
  be overwritten with left and right eigenvectors.

  If the QR algorithm failed to compute all the eigenvalues, throws ExceptionInfo, with the information
  on the index of the first eigenvalue that converged.
  If `w` is not column-oriented, or has less than 2 columns, throws ExceptionInfo.
  If `vl` or `vr` dimensions do not fit with `a`'s dimensions, throws ExceptionInfo.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info about [lapacke_?geev](https://software.intel.com/en-us/node/521147).
  "
  ([^Matrix a ^Matrix w ^Matrix vl ^Matrix vr]
   (if (and (= (.mrows a) (.ncols a))
            (= (.mrows a) (.mrows w)) (< 1 (.ncols w))
            (or (nil? vl) (and (= (.mrows a) (.mrows vl) (.ncols vl)) (api/fits-navigation? a vl)))
            (or (nil? vr) (and (= (.mrows a) (.mrows vr) (.ncols vr)) (api/fits-navigation? a vr))))
     (api/ev (api/engine a) a w vl vr)
     (throw (ex-info "You cannot compute eigenvalues of a non-square matrix or with the provided destinations."
                     {:a (str a) :w (str w) :vl (str vl) :vr (str vr) :errors
                      (cond-into []
                                 (not (= (.mrows a) (.ncols a))) "a is not a square matrix"
                                 (not (= (.mrows a) (.mrows w))) "a and w have different row dimensions"
                                 (not (< 1 (.ncols w))) "w has less than 2 columns"
                                 (not (or (nil? vl) (= (.mrows a) (.mrows vl) (.ncols vl))))
                                 "a and vl dimensions do not fit"
                                 (not (or (nil? vr) (= (.mrows a) (.mrows vr) (.ncols vr))))
                                 "a and vr dimensions do not fit"
                                 (not (api/fits-navigation? a vl)) "a and vl do not have the same orientation"
                                 (not (api/fits-navigation? a vr)) "a and vr do not have the same orientation")}))))
  ([a w]
   (ev! a w nil nil))
  ([^Matrix a vl vr]
   (let-release [w (ge (api/factory a) (.mrows a) 2)]
     (ev! a w vl vr)))
  ([^Matrix a]
   (ev! a nil nil)))

(defn svd!
  "Computes the singular value decomposition of a matrix `a`.

  On exit, `a` is destroyed, or, if `u` or `vt` are `nil`, overwritten with U or transposed V
  singular vectors of `a`. `s` is populated with sorted singular values. If the factorization
  does not converge, `superb` is populated  with the unconverged superdiagonal elements
  (see LAPACK documentation). If called without `u` and `vt`, U and transposed V are not computed.

  If the reduction to bidiagonal form failed to converge, throws ExceptionInfo, with the information
  on the number of converged superdiagonals.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info about [lapacke_?gesvd](https://software.intel.com/en-us/node/521150).
  "
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
       (throw (ex-info "You can not do a singular value decomposition with incompatible or ill-fitting arguments."
                     {:a (str a) :s (str s) :u (str u) :vt (str vt) :superb (str superb) :errors
                      (cond-into []
                                 (not (= min-mn (.dim s))) "dimension of s is no (min m n)"
                                 (not (= min-mn (.dim superb))) "dimension of superb is no (min m n)"

                                 (not (or (nil? u) (= m (.mrows u) (.ncols u)))) "u is not a mxm matrix"
                                 (not (or (nil? u) (= min-mn (.ncols u)))) "ncols of u is not equals (min m n)"
                                 (not (or (nil? u) (= n (.mrows u)))) "mrows of vt is not equals n"
                                 (not (or (nil? u) (api/fits-navigation? a u))) "a and u do not have the same orientation"
                                 (not (or (nil? vt) (api/fits-navigation? a vt))) "a and vt do not have the same orientation"
                                 (not (or (nil? vt) (= n (.mrows vt) (.ncols vt)))) "vt is not a nxn matrix"
                                 (not (or (nil? vt) (= min-mn (.mrows vt)))) "mrows of vt is not equals (min m n)"
                                 (not (or (nil? vt) (= n (.ncols vt)))) "ncols of vt is not equals n")})))))
  ([^Matrix a ^Vector s ^Vector superb]
   (if (and (= (min (.mrows a) (.ncols a)) (.dim s) (.dim superb)))
     (api/svd (api/engine a) a s superb)
     (throw (ex-info "You can not do a singular value decomposition with incompatible or ill-fitting arguments."
                     {:a (str a) :s (str s) :superb (str superb) :errors
                      (cond-into []
                                 (not (= min-mn (.dim s))) "dimension of s is no (min m n)"
                                 (not (= min-mn (.dim superb))) "dimension of superb is no (min m n)")})))))
