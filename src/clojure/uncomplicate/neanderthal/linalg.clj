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
  - [Linear Equation Computational Routines](https://software.intel.com/en-us/mkl-developer-reference-c-lapack-linear-equation-computational-routines)
  - [Linear Equations](https://software.intel.com/en-us/mkl-developer-reference-c-lapack-linear-equation-routines)
  - [Orthogonal Factorizations (Q, R, L)](https://software.intel.com/en-us/node/521003)
  - [Singular Value Decomposition](https://software.intel.com/en-us/node/521036)
  - [Symmetric Eigenvalue Problems](https://software.intel.com/en-us/node/521119)
  - Other LAPACK documentation, as needed.
  "
  (:require [uncomplicate.commons
             [core :refer [let-release with-release]]
             [utils :refer [cond-into]]]
            [uncomplicate.neanderthal.core :refer [vctr vctr? ge copy]]
            [uncomplicate.neanderthal.internal
             [api :as api]
             [common :refer [dragan-says-ex]]])
  (:import [uncomplicate.neanderthal.internal.api Vector Matrix Changeable]))

;; ============================= LAPACK =======================================================

;; =============  Triangular Linear Systems LAPACK ============================================

(defn trf!
  "Triangularizes a non-triangular matrix `a`. Destructively computes the LU (or LDLt, or UDUt, or GGt)
  factorization of a `mxn` matrix `a`,
  and places it in a record that contains `:lu` and `:ipiv`.

  Overwrites `a` with L and U. L is stored as a lower unit triangle, and U as an upper triangle.
  Pivot is  placed into the `:ipiv`, a vector of **integers or longs**.

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [lapacke_?getrf](https://software.intel.com/en-us/mkl-developer-reference-c-getrf).
  "
  [^Matrix a]
  (api/create-trf (api/engine a) a false))

(defn ptrf!
  "Triangularizes a positive definite symmetric matrix `a`. Destructively computes
  the Cholesky factorization of a `nxn` matrix `a`, and places it in a record with the key `gg`.

  Overwrites `a` with G in the lower triangle, or G transposed in the upper triangle,
  depending on whether `a` is `:lower` or `:upper`. Cholesky does not need pivoting.

  If `a` is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [lapacke_?potrf](https://software.intel.com/en-us/mkl-developer-reference-c-potrf).
  "
  [^Matrix a]
  (api/create-ptrf (api/engine a) a))

(defn trf
  "Triangularizes a non-TR matrix `a`. Computes the LU (or LDLt, or UDUt)
  factorization of a `mxn` matrix `a`, and places it in a record that contains `:lu`, `:a` and `:ipiv`.
  If `a` is positive definite symmetric, compute the Cholesky factorization, that contains
  only `:gg` and no ipiv is needed.

  Pivot is placed into the `:ipiv`, a vector of **integers or longs** (if applicable).

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [[trf!]] and [[ptrf!]].
  "
  [^Matrix a]
  (api/create-trf (api/engine a) a true))

(defn tri!
  "Destructively computes the inverse of a triangularized matrix `a`.
  Overwrites `a` (or `:lu`) with a `nxn` inverse.

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.
  If the matrix is not square, throws ExceptionInfo.

  See related info about [lapacke_?getri](https://software.intel.com/en-us/mkl-developer-reference-c-getri).
  "
  [^Matrix a]
  (if (= (.mrows a) (.ncols a))
    (api/trtri! a)
    (throw (ex-info "Can not compute an inverse of a non-square matrix."
                    {:m (.mrows a) :n (.ncols a)}))))

(defn tri
  "Computes the inverse of a triangularized matrix `a`. Returns the results in a new matrix instance.

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.
  If the matrix is not square, throws ExceptionInfo.

  See related info about [[tri!]].
  "
  [^Matrix a]
  (if (= (.mrows a) (.ncols a))
    (api/trtri a)
    (throw (ex-info "Can not compute an inverse of a non-square matrix."
                    {:m (.mrows a) :n (.ncols a)}))))

(defn trs!
  "Destructively solves a system of linear equations with a triangularized matrix `a`,
  with multiple right hand sides matrix `b`. Overwrites `b` by the solution matrix.

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [https://software.intel.com/en-us/mkl-developer-reference-c-getrs).
  "
  [^Matrix a ^Matrix b]
  (if (and (api/compatible? a b) (= (.ncols a) (.mrows b)))
    (api/trtrs! a b)
    (throw (ex-info "Dimensions and orientation of a and b do not fit."
                    {:a  (api/info a) :b (api/info b) :errors
                     (cond-into []
                                (not (api/compatible? a b)) "incompatible a and b"
                                (not (= (.ncols a) (.mrows b))) "a and b dimensions do not fit")}))))

(defn trs
  "Solves a system of linear equations with a triangularized matrix `a`,
  with multiple right hand sides matrix `b`. Returns the results in a new matrix instance.

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [[trs!]].
  "
  [^Matrix a ^Matrix b]
  (if (and (api/compatible? a b) (= (.ncols a) (.mrows b)))
    (api/trtrs a b)
    (throw (ex-info "Dimensions and orientation of a and b do not fit."
                    {:a  (api/info a) :b (api/info b) :errors
                     (cond-into []
                                (not (api/compatible? a b)) "incompatible a and b"
                                (not (= (.ncols a) (.mrows b))) "a and b dimensions do not fit")}))))

(defn sv!
  "Destructively solves a system of linear equations with a square coefficient matrix `a`,
  and multiple right  hand sides matrix `b`. Overwrites `b` by the solution matrix.

  Overwrites `a` with L and U, and `b` with the solution.
  L is stored as a lower unit triangle, and U as an upper triangle. Pivot is not retained.
  If you need to reuse LU, use [[trf]], and [[trs]], or their destructive versions.

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [lapacke_?gesv](https://software.intel.com/en-us/mkl-developer-reference-c-gesv).
  "
  [^Matrix a ^Matrix b]
  (if (and (api/compatible? a b) (and (= (.ncols a) (.mrows b))))
    (api/sv (api/engine a) a b false)
    (dragan-says-ex "You cannot solve incompatible or ill-fitting structures."
                    {:a  (api/info a) :b (api/info b) :errors
                     (cond-into []
                                (not (api/compatible? a b)) "incompatible a and b"
                                (not (= (.ncols a) (.mrows b))) "a and b dimensions do not fit")})))

(defn psv!
  "Destructively solves a system of linear equations with a positive definite symmetric coefficient
  matrix `a`, and multiple right sides matrix `b`. Overwrites `b` by the solution matrix.

  Overwrites `a` with G, and `b` with the solution.

  If G is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [lapacke_?gesv](https://software.intel.com/en-us/mkl-developer-reference-c-gesv).
  "
  [^Matrix a ^Matrix b]
  (if (and (api/compatible? a b) (and (= (.ncols a) (.mrows b))))
    (api/sv (api/engine a) a b)
    (dragan-says-ex "You cannot solve incompatible or ill-fitting structures."
                    {:a  (api/info a) :b (api/info b) :errors
                     (cond-into []
                                (not (api/compatible? a b)) "incompatible a and b"
                                (not (= (.ncols a) (.mrows b))) "a and b dimensions do not fit")})))

(defn sv
  "Solves a system of linear equations with a square coefficient matrix `a`,
  and multiple right  hand sides matrix `b`. Returns the results in a new matrix instance.

  Overwrites `a` with L and U, and `b` with the solution.
  L is stored as a lower unit triangle, and U as an upper triangle. Pivot is not retained.
  If you need to reuse LU, use [[trf]], and [[trs]], or their destructive versions.

  If `a` is symmetric, tries to do Cholesky factorization first, and only does LDLt if
  it turns out not to be positive definite.

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [[sv!]].
  "
  [a ^Matrix b]
  (let-release [res (ge (api/factory b) (.mrows b) (.ncols b))]
    (api/copy (api/engine b) b res)
    (api/sv (api/engine a) a res true)
    res))

(defn con
  "Computes the reciprocal of the condition number of a triangularized matrix `lu`.

  If `nrm1?` is true, computes the reciprocal based on 1 norm, if not, on infinity norm.
  `nrm` and matching `nrm1?` have to be supplied.

  If the LU has been stale, or norm is not possible to compute, throws ExceptionInfo.

  See related info about [lapacke_?gecon] (https://software.intel.com/en-us/mkl-developer-reference-c-gecon).
  "
  (^double [^Matrix lu nrm nrm1?]
   (api/trcon lu nrm nrm1?))
  (^double [^Matrix a nrm1-?]
   (if (number? nrm1-?)
     (api/trcon a nrm1-? true)
     (api/trcon a nrm1-?)))
  (^double [^Matrix a]
   (api/trcon a true)))

(defn det
  "Computes the determinant of a triangularized matrix `a`.

  If the matrix is not square, throws ExceptionInfo."
  [^Matrix a]
  (if (= (.mrows a) (.ncols a))
    (api/trdet a)
    (throw (ex-info "Determinant computation requires a square matrix."
                    {:mrows (.mrows a) :ncols (.ncols a)}))))

;; =============== Orthogonal Factorization (L, Q, R) LAPACK =======================================

(defn ^:private min-mn ^long [^Matrix a]
  (max 1 (min (.mrows a) (.ncols a))))

(defmacro ^:private with-lqrf-check [a tau expr]
  `(if (and (= (.dim ~tau) (min-mn ~a)) (api/compatible? ~a ~tau))
     ~expr
     (throw (ex-info "Dimension of tau is not equals to (min ncols mrows) of a."
                     {:m (.mrows ~a) :n (.ncols ~a) :dim (.dim ~tau)}))))

(defmacro ^:private with-gq*-check [a tau expr]
  `(if (and (<= 1 (.dim ~tau) (.ncols ~a) (.mrows ~a)) (api/compatible? ~a ~tau))
     ~expr
     (throw (ex-info "Dimensions of a and tau do not fit requirements for computing Q."
                     {:m (.mrows ~a) :n (.ncols ~a) :dim (.dim ~tau) :errors
                      (cond-into []
                                 (not (<= (.dim ~tau) (.ncols ~a))) "(not (<= (dim tau) (ncols a)))"
                                 (not (<= (.ncols ~a) (.mrows ~a))) "(not (<= (ncols ~a) (mrows a)))")}))))

(defmacro ^:private with-g*q-check [a tau expr]
  `(if (and (<= (.dim ~tau) (.mrows ~a) (.ncols ~a)) (api/compatible? ~a ~tau))
     ~expr
     (throw (ex-info "Dimensions of a and tau do not fit requirements for computing Q."
                     {:m (.mrows ~a) :n (.ncols ~a) :dim (.dim ~tau) :errors
                      (cond-into []
                                 (not (<= (.dim ~tau) (.mrows ~a))) "(not (<= (dim tau) (mrows a)))"
                                 (not (<= (.mrows ~a) (.ncols ~a))) "(not (<= (mrows a) (ncols a)))")}))))

(defmacro ^:private with-mq*-check [a tau c left expr]
  `(let [r# (long (if ~left (.mrows ~c) (.ncols ~c)))]
     (if (and (<= (.dim ~tau) r#) (= r# (.mrows ~a)) (= (.ncols ~a) (.dim ~tau))
              (api/compatible? ~a ~tau))
       ~expr
       (throw (ex-info "Dimensions of a, tau, and c do not fit matrix multiplication."
                       {:a (api/info ~a) :c (api/info ~c) :tau (api/info ~tau) :errors
                        (cond-into []
                                   (not (<= (.dim ~tau) r#))
                                   (format "(not (<= (dim tau) %s))" (if ~left "(mrows c)" "(ncols c)"))
                                   (not (= r# (.mrows ~a)))
                                   (format "(not (= %s (mrows a)))" (if ~left "(mrows c)" "(ncols c)"))
                                   (not (= (.ncols ~a) (.dim ~tau))) "(not (= (ncols a) (dim tau)))")})))))

(defmacro ^:private with-m*q-check [a tau c left expr]
  `(let [r# (long (if ~left (.mrows ~c) (.ncols ~c)))]
     (if (and (<= (.dim ~tau) r#) (= (.mrows ~c) (.ncols ~a)) (= (.mrows ~a) (.dim ~tau))
              (api/compatible? ~a ~tau))
       ~expr
       (throw (ex-info "Dimensions of a, tau, and c do not fit matrix multiplication."
                       {:a (api/info ~a) :c (api/info ~c) :tau (api/info ~tau) :errors
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
                    {:a (api/info a) :b (api/info b) :errors
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
            (or (nil? vl) (and (= (.mrows a) (.mrows vl) (.ncols vl))
                               (api/compatible? a vl) (api/fits-navigation? a vl)))
            (or (nil? vr) (and (= (.mrows a) (.mrows vr) (.ncols vr))
                               (api/compatible? a vr)(api/fits-navigation? a vr))))
     (api/ev (api/engine a) a w vl vr)
     (throw (ex-info "You cannot compute eigenvalues of a non-square matrix or with the provided destinations."
                     {:a (api/info a) :w (api/info w) :vl (api/info vl) :vr (api/info vr) :errors
                      (cond-into []
                                 (not (= (.mrows a) (.ncols a))) "a is not a square matrix"
                                 (not (= (.mrows a) (.mrows w))) "a and w have different row dimensions"
                                 (not (< 1 (.ncols w))) "w has less than 2 columns"
                                 (not (or (nil? vl) (= (.mrows a) (.mrows vl) (.ncols vl))))
                                 "a and vl dimensions do not fit"
                                 (not (or (nil? vr) (= (.mrows a) (.mrows vr) (.ncols vr))))
                                 "a and vr dimensions do not fit"
                                 (and vl (not (api/fits-navigation? a vl))) "a and vl do not have the same orientation"
                                 (and vr (not (api/fits-navigation? a vr))) "a and vr do not have the same orientation")}))))
  ([a w]
   (ev! a w nil nil))
  ([^Matrix a vl vr]
   (let-release [w (ge (api/factory a) (.mrows a) 2)]
     (ev! a w vl vr)))
  ([^Matrix a]
   (ev! a nil nil)))

;; ================= Singular Value Decomposition ================================================

(defn svd!
  "Computes the singular value decomposition of a matrix `a`.

  On exit, `a` is destroyed, or, if `u` or `vt` are `nil`, overwritten with U or transposed V
  singular vectors of `a`. `s` is a diagonal banded matrix populated with sorted singular values.
  If the factorization does not converge, a diagonal banded matrix `superb` is populated with
  the unconverged superdiagonal elements (see LAPACK documentation). If called without `u` and `vt`,
  U and transposed V are not computed.

  If the reduction to bidiagonal form failed to converge, throws ExceptionInfo, with the information
  on the number of converged superdiagonals.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info about [lapacke_?gesvd](https://software.intel.com/en-us/node/521150).
  "
  ([^Matrix a ^Matrix sigma ^Matrix u ^Matrix vt ^Matrix superb]
   (let [m (.mrows a)
         n (.ncols a)
         min-mn (min m n)]
     (if (and (or (nil? u) (and (or (= m (.mrows u) (.ncols u))
                                    (and (= m (.mrows u)) (= min-mn (.ncols u))))
                                (api/compatible? a u) (api/fits-navigation? a u)))
              (or (nil? vt) (and (or (= n (.mrows vt) (.ncols vt))
                                     (and (= min-mn (.mrows vt)) (= n (.ncols vt))))
                                 (api/compatible? a vt) (api/fits-navigation? a vt)))
              (= m (.mrows sigma) (.mrows superb)) (= n (.ncols sigma) (.ncols superb)))
       (api/svd (api/engine a) a sigma u vt superb)
       (throw (ex-info "You can not do a singular value decomposition with incompatible or ill-fitting arguments."
                       {:a (api/info a) :s (api/info sigma) :u (api/info u) :vt (api/info vt) :superb (api/info superb)
                        :errors
                        (cond-into []
                                   (not (= m (.mrows sigma))) "mrows of sigma is not m"
                                   (not (= n (.ncols sigma))) "ncols of sigma is not n"
                                   (not (= m (.mrows superb))) "mrows of superb is not m"
                                   (not (= n (.ncols superb))) "ncols of superb is not n"
                                   (not (or (nil? u) (= m (.mrows u) (.ncols u)))) "u is not a mxm matrix"
                                   (not (or (nil? u) (= min-mn (.ncols u)))) "ncols of u is not equals (min m n)"
                                   (not (or (nil? u) (= n (.mrows u)))) "mrows of vt is not equals n"
                                   (not (or (nil? u) (api/fits-navigation? a u))) "a and u do not have the same orientation"
                                   (not (or (nil? vt) (api/fits-navigation? a vt))) "a and vt do not have the same orientation"
                                   (not (or (nil? vt) (= n (.mrows vt) (.ncols vt)))) "vt is not a nxn matrix"
                                   (not (or (nil? vt) (= min-mn (.mrows vt)))) "mrows of vt is not equals (min m n)"
                                   (not (or (nil? vt) (= n (.ncols vt)))) "ncols of vt is not equals n")})))))
  ([^Matrix a ^Matrix sigma ^Matrix superb]
   (if (and (= (.mrows a) (.mrows sigma) (.mrows superb)) (= (.ncols a) (.ncols sigma) (.ncols superb))
            (api/compatible? a sigma) (api/compatible? a superb))
     (api/svd (api/engine a) a sigma superb)
     (throw (ex-info "You can not do a singular value decomposition with incompatible or ill-fitting arguments."
                     {:a (api/info a) :s (api/info sigma) :superb (api/info superb) :errors
                      (cond-into []
                                 (not (= (.mrows a) (.mrows sigma))) "mrows of sigma is not m"
                                 (not (= (.ncols a) (.ncols sigma))) "ncols of sigma is not n"
                                 (not (= (.mrows a) (.mrows superb))) "mrows of superb is not m"
                                 (not (= (.ncols a) (.ncols superb))) "ncols of superb is not n")})))))
