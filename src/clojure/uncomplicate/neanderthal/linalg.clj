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

  - Linear equations and LU factorization: [[trf!]], [[trf]], [[ptrf!]], [[tri!]], [[tri]], [[trs!]],
  [[trs]], [[sv!]], [[sv]], [[psv!]], [[psv]], [[con]], [[det]],.
  - Orthogonal factorizations: [[qrf!]], [[qrf]], [[qrfp!]], [[qrfp]], [[qpf!]], [[qpf]], [[qrfp!]],
  [[qrfp]], [[rqf!]], [[rqf]], [[qlf!]], [[qlf]], [[lqf!]], [[qlf!]], [[qlf]], [[org!]], [[org]],
  - Linear least squares: [[ls!]], [[ls]], [[lse!]], [[lse]], [[gls!]], [[gls]].
  - Eigen decomposition: [[ev!]], [[es!]].
  - Singular value decomposition (SVD): [[svd!]], [[svd]].

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
             [core :refer [let-release with-release release info]]
             [utils :refer [cond-into dragan-says-ex]]]
            [uncomplicate.neanderthal.core :refer [vctr vctr? ge copy gd raw symmetric?]]
            [uncomplicate.neanderthal.internal
             [api :as api]
             [common :refer [->SVDecomposition qr-factorization qp-factorization
                             rq-factorization ql-factorization lq-factorization]]])
  (:import [uncomplicate.neanderthal.internal.api Vector Matrix Changeable]))

;; ===================================== Linear Systems ============================================

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
  (api/create-trf a false))

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
  (api/create-ptrf a))

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
  (api/create-trf a true))

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
    (dragan-says-ex "You can not compute an inverse of a non-square matrix."
                    {:m (.mrows a) :n (.ncols a)})))

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
    (dragan-says-ex "You can not compute an inverse of a non-square matrix."
                    {:m (.mrows a) :n (.ncols a)})))

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
    (dragan-says-ex "Dimensions and orientation of a and b do not fit."
                    {:a  (info a) :b (info b) :errors
                     (cond-into []
                                (not (api/compatible? a b)) "incompatible a and b"
                                (not (= (.ncols a) (.mrows b))) "a and b dimensions do not fit")})))

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
    (dragan-says-ex "Dimensions and orientation of a and b do not fit."
                    {:a  (info a) :b (info b) :errors
                     (cond-into []
                                (not (api/compatible? a b)) "incompatible a and b"
                                (not (= (.ncols a) (.mrows b))) "a and b dimensions do not fit")})))

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
                    {:a  (info a) :b (info b) :errors
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
                    {:a  (info a) :b (info b) :errors
                     (cond-into []
                                (not (api/compatible? a b)) "incompatible a and b"
                                (not (= (.ncols a) (.mrows b))) "a and b dimensions do not fit")})))

(defn sv
  "Solves a system of linear equations with a square coefficient matrix `a`,
  and multiple right  hand sides matrix `b`. Returns the solution in a new matrix instance.

  If `a` is symmetric, tries to do Cholesky factorization first, and only does LDLt if
  it turns out not to be positive definite.

  If U is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [[sv!]].
  "
  [a ^Matrix b]
  (let-release [b-copy (api/create-ge (api/factory b) (.mrows b) (.ncols b) true false)]
    (api/copy (api/engine b) b b-copy)
    (api/sv (api/engine a) a b-copy true)))

(defn psv
  "Solves a system of linear equations with a positive definite symmetric coefficient
  matrix `a`, and multiple right sides matrix `b`. Returns the solution matrix in a new instance.

  If G is exactly singular (it can't be used for solving a system of linear equations),
  throws ExceptionInfo.

  See related info about [[psv!]].
  "
  [a ^Matrix b]
  (let-release [b-copy (api/create-ge (api/factory b) (.mrows b) (.ncols b) true false)]
    (api/copy (api/engine b) b b-copy)
    (psv! a b-copy)))

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
    (dragan-says-ex "Determinant computation requires a square matrix."
                    {:mrows (.mrows a) :ncols (.ncols a)})))

;; ================================== Least Squares  =============================================

(defn qrf!
  "Destructively computes the QR factorization of a `m x n` matrix and places it in record
  that contains `:or` and `:tau`.

  The input is a matrix `a`. The output overwrites the contents of `a`. Output QR is laid out
  in `a` in the following way: The elements in the upper triangle (or trapezoid) contain the
  `(min m n) x n` upper triangular (or trapezoidal) matrix R. The elements in the lower triangle
  (or trapezoid) **below the diagonal**, with the vector `tau` contain Q as a product of `(min m n)`
  elementary reflectors. **Other routines work with Q in this representation**. If you need
  to compute q, call [[org!]] or [[org]].

  See related info about [lapacke_?geqrf](https://software.intel.com/en-us/node/521004).
  "
  [a]
  (qr-factorization a false api/qrf))

(defn qrf
  "Purely computes the QR factorization of a GE `m x n` matrix and places it in record
  that contains `:or` and `:tau`.

  See [[qrf!]]"
  [^Matrix a]
  (let-release [a-copy (copy a)]
    (qr-factorization a-copy true api/qrf)))

(defn qpf!
  "Destructively computes the QR factorization with pivoting of a `m x n` matrix and places it in record
  that contains `:or` and `:tau`.

  It is similar to [[qrf!]] and can replace it, with the caveat that the results have swapped columns
  (because of pivoting) so the solutions based on it have to be permuted back.

  See related info about [lapacke_?geqrf](https://software.intel.com/en-us/node/521004).
  "
  [a]
  (qp-factorization a false))

(defn qpf
  "Purely computes the QR factorization of a GE `m x n` matrix and places it in record
  that contains `:or` and `:tau`.

  It is similar to [[qrf!]] and can replace it, with the caveat that the results have swapped columns
  (because of pivoting) so the solutions based on it have to be permuted back.

  See [[qrf!]]"
  [^Matrix a]
  (let-release [a-copy (copy a)]
    (qp-factorization a-copy true)))

(defn qrfp!
  "Destructively computes the QR factorization of a GE `m x n` matrix, with non-negative diagonal elements.

  See [[qrf!]].
  See related info about [lapacke_?geqrfp](https://software.intel.com/en-us/node/468946).
  "
  [a]
  (qr-factorization a false api/qrfp))

(defn qrfp
  "Purely computes the QR factorization of a GE `m x n` matrix, with non-negative diagonal elements.

  See [[qrf!]].
  See related info about [lapacke_?geqrfp](https://software.intel.com/en-us/node/468946).
  "
  [^Matrix a]
  (let-release [a-copy (copy a)]
    (qr-factorization a-copy true api/qrfp)))

(defn rqf!
  "Destructively computes the RQ factorization of a GE `m x n`.

  See [[qrf!]].

  See related info about [lapacke_?gerqf](https://software.intel.com/en-us/node/521024).
  "
  [a]
  (rq-factorization a false))

(defn rqf
  "Computes the RQ factorization of a GE `m x n`.

  See [[qrf!]].

  See related info about [lapacke_?gerqf](https://software.intel.com/en-us/node/521024).
  "
  [^Matrix a]
  (let-release [a-copy (copy a)]
    (rq-factorization a-copy true)))

(defn qlf!
  "Destructively computes the QL factorization of a GE `m x n` matrix.

  See [[qrf!]].

  See related info about [lapacke_?geqlf](https://software.intel.com/en-us/node/521019).
  "
  [a]
  (ql-factorization a false))

(defn qlf
  "Computes the QL factorization of a GE `m x n` matrix.

  See [[qrf!]].

  See related info about [lapacke_?geqlf](https://software.intel.com/en-us/node/521019).
  "
  [^Matrix a]
  (let-release [a-copy (copy a)]
    (ql-factorization a-copy true)))

(defn lqf!
  "Destructively computes the LQ factorization of a GE `m x n` matrix.

  See [[qrf!]].

  See related info about [lapacke_?gelqf](https://software.intel.com/en-us/node/521014).
  "
  [a]
  (lq-factorization a false))

(defn lqf
  "Computes the LQ factorization of a GE `m x n` matrix.

  See [[qrf!]].

  See related info about [lapacke_?gelqf](https://software.intel.com/en-us/node/521014).
  "
  [^Matrix a]
  (let-release [a-copy (copy a)]
    (lq-factorization a-copy true)))

(defn org!
  "Destructively generates the real orthogonal matrix Q of the QR, RQ, QL or LQ factorization formed by
  [[qrf!]] or [[qrfp!]] (or [[rqf!]] etc.).

  The input is a structure containing orthogonalized matrix `:lu` and vector `:tau` that were
  previously processed by [[qrf!]] (and friends). Overwrites the input with the appropriate
  portion of the resulting matrix Q.

  See related info about [lapacke_?orgqr](https://software.intel.com/en-us/node/468956).
  "
  [or]
  (api/org! or))

(defn org
  "Generates the real orthogonal matrix Q of the QR, RQ, QL or LQ factorization formed by
  [[qrf!]] or [[qrfp!]] (or [[rqf!]] etc.).

  The input is a structure containing orthogonalized matrix `:lu` and vector `:tau` that were
  previously processed by [[qrf!]] (and friends).

  See related info about [lapacke_?orgqr](https://software.intel.com/en-us/node/468956).
  "
  [or]
  (api/org or))

(defn ls!
  "Destructively solves an overdetermined or underdetermined linear system `AX = B` with full rank
  matrix using QR or LQ factorization.

  Overwrites `a` with the factorization data:
  - QR if `m >= n`;
  - LQ if `m < n`.

  Overwrites b with:
  - the least squares solution vectors if `m >= n`
  - minimum norm solution vectors if `m < n`.

  If `a` and `b` do not have the same layout (column or row oriented), throws ExceptionInfo.
  If the least squares cannot be computed the function throws ExceptionInfo.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info about [lapacke_?gels](https://software.intel.com/en-us/node/521112).
  "
  [^Matrix a ^Matrix b]
  (if (and (<= (max 1 (.mrows a) (.ncols a)) (.mrows b))
           (api/compatible? a b) (api/fits-navigation? a b))
    (api/ls (api/engine a) a b)
    (throw (ex-info "You cannot solve least squares described by incompatible or ill-fitting matrices."
                    {:a (info a) :b (info b) :errors
                     (cond-into []
                                (not (<= (max 1 (.mrows a) (.ncols a)) (.mrows b)))
                                "dimensions of a and b do not fit"
                                (not (api/compatible? a b)) "a and b do are not compatible"
                                (not (api/fits-navigation? a b)) "a and b do not have the same layout")}))))

(defn ls
  "Solves an overdetermined or underdetermined linear system `ax=b` with full rank matrix using
  QR or LQ factorization.

  Uses a temporary copy of`a` for the factorization data:
  - QR if `m >= n`;
  - LQ if `m < n`.

  Returns a new instance containing:
  - the least squares solution vectors if `m >= n`
  - minimum norm solution vectors if `m < n`.

  If `a` and `b` do not have the same layout (column or row oriented), throws ExceptionInfo.
  If the least squares cannot be computed the function throws ExceptionInfo.
  If some value in the native call is illegal, throws ExceptionInfo.

  See [[ls!]].
  See related info about [lapacke_?gels](https://software.intel.com/en-us/node/521112).
  "
  [^Matrix a ^Matrix b]
  (with-release [a-copy (copy a)]
    (let-release [b-copy (copy b)]
      (ls! a-copy b-copy))))

(defn lse!
  "Destructively solves the generalized linear least squares problem with equality constraints using RQ factorization.

  Minimizes the 2-norm `||ax-c||` subject to constraints `bx=d`.

  Overwrites `a` with the T ([[rqf]]), `x` with the solution, b with R ([[rqf]]), d with garbage,
  c with the residual sum of squares.

  If the dimensions of arguments do not fit, throws ExceptionInfo.
  If `a` and `b` do not have the same layout (column or row oriented), throws ExceptionInfo.
  If the least squares cannot be computed the function throws ExceptionInfo.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info at [lapacke_?gglse](https://software.intel.com/en-us/mkl-developer-reference-c-gglse).
  "
  [^Matrix a ^Matrix b ^Vector c ^Vector d ^Vector x]
  (let [m (.mrows a)
        n (.ncols a)
        p (.mrows b)]
    (if (and (<= 0 p n m (+ m p)) (= n (.ncols b)) (<= m (.dim c)) (<= p (.dim d)) (<= n (.dim x))
             (api/compatible? a b) (api/compatible? a c) (api/compatible? a d) (api/compatible? a x)
             (api/fits-navigation? a b))
      (api/lse (api/engine a) a b c d x)
      (throw (ex-info "You cannot solve least squares with equality constraints described by incompatible or ill-fitting structures."
                      {:a (info a) :b (info b) :c (info c) :d (info d) :errors
                       (cond-into []
                                  (not (<= 0 p n m (+ m p))) "dimensions of a and b do not fit"
                                  (not (= n (.ncols b))) "a and be should have the same number of columns"
                                  (not (<= m (.dim c))) "dimension of c should be at least number of rows of a"
                                  (not (<= p (.dim d))) "dimension of d should be at least number of rows of b"
                                  (not (api/compatible? a b)) "a and b are not compatible"
                                  (not (api/compatible? a c)) "a and c are not compatible"
                                  (not (api/compatible? a d)) "a and d are not compatible"
                                  (not (api/compatible? a x)) "a and x are not compatible"
                                  (not (api/fits-navigation? a b)) "a and b do not have the same layout")})))))

(defn lse
  "Solves the generalized linear least squares problem with equality constraints using RQ factorization.

  See [[lse!]]"
  [^Matrix a b c d]
  (with-release [a-copy (copy a)
                 b-copy (copy b)
                 c-copy (copy c)
                 d-copy (copy d)]
    (let-release [x (api/create-vector (api/factory a) (.ncols a) false)]
      (lse! a-copy b-copy c-copy d-copy x))))

(defn gls!
  "Destructively solves the generalized linear least squares problem using a generalized QR factorization.

   Minimizes the 2-norm `||y||` subject to constraints `d=ax+by`.

  Overwrites `a` with the R ([[rqf]]), `x` with the solution, b with T ([[rqf]]), d with garbage,
  x ad y with solution to the GLS problem.

  If the dimensions of arguments do not fit, throws ExceptionInfo.
  If `a` and `b` do not have the same layout (column or row oriented), throws ExceptionInfo.
  If the least squares cannot be computed the function throws ExceptionInfo.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info at [lapacke_?gglse](https://software.intel.com/en-us/mkl-developer-reference-c-ggglm).
  "
  [^Matrix a ^Matrix b ^Vector d ^Vector x ^Vector y]
  (let [m (.mrows a)
        n (.ncols a)
        p (.ncols b)]
    (if (and (<= 0 (- m n) p (+ m p))(= m (.mrows b)) (<= m (.dim d)) (<= n (.dim x)) (<= p (.dim y))
             (api/compatible? a b) (api/compatible? a d) (api/compatible? a x) (api/compatible? a y)
             (api/fits-navigation? a b))
      (do
        (api/gls (api/engine a) a b d x y)
        [x y])
      (throw (ex-info "You cannot solve generalized least squares described by incompatible or ill-fitting structures."
                      {:a (info a) :b (info b) :d (info d) :x (info x) :y (info y) :errors
                       (cond-into []
                                  (not (<= 0 (- m n) p (+ m p))) "dimensions of a and b do not fit"
                                  (not (= m (.mrows b))) "a and be should have the same number of rows"
                                  (not (<= m (.dim d))) "dimension of d should be at least number of rows of a"
                                  (not (<= n (.dim x))) "dimension of x should be at least number of columns of a"
                                  (not (<= p (.dim y))) "dimension of y should be at least number of columns of b"
                                  (not (api/compatible? a b)) "a and b are not compatible"
                                  (not (api/compatible? a d)) "a and d are not compatible"
                                  (not (api/compatible? a x)) "a and x are not compatible"
                                  (not (api/compatible? a y)) "a and y are not compatible"
                                  (not (api/fits-navigation? a b)) "a and b do not have the same layout")})))))

(defn gls
  "Solves the generalized linear least squares problem using a generalized QR factorization.

   Minimizes the 2-norm `||y||` subject to constraints `d=ax+by`.

  See [[gls!]].
  "
  [^Matrix a ^Matrix b d]
  (with-release [a-copy (copy a)
                 b-copy (copy b)
                 d-copy (copy d)]
    (let-release [x (api/create-vector (api/factory a) (.ncols a) false)
                  y (api/create-vector (api/factory a) (.ncols b) false)]
      (gls! a-copy b-copy d-copy x y))))

;; =============================== Eigenproblems =================================================

(defn ev!
  "Computes the eigenvalues and left and right eigenvectors of a matrix `a`.

  `a` - source matrix and computed orthogonal factorization (m x m).
  `w` - computed eigenvalues (m x 2 or k x 1 for symmetric matrices).
  `vl` and/or `vr` - left and right eigenvectors (m x m or m x k for symmetric matrices).

  On exit, `a` is overwritten with QR factors. The first 2 columns of a column-oriented GE matrix
  `w` are overwritten with eigenvalues of `a`. If `vl` and `vr` GE matrices are provided, they will
  be overwritten with left and right eigenvectors. For symmetric matrices, computes the first k
  eigenvalues (k <= m). If `vl` and/or `vr` are nil, eigenvectors are not computed.

  If the QR algorithm failed to compute all the eigenvalues, throws ExceptionInfo, with the information
  on the index of the first eigenvalue that converged.
  If `w` is not column-oriented, or does not have 2 columns, throws ExceptionInfo.
  If `vl` or `vr` dimensions do not fit with `a`'s dimensions, throws ExceptionInfo.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info about [lapacke_?geev](https://software.intel.com/en-us/node/521147).
  "
  ([^Matrix a ^Matrix w ^Matrix vl ^Matrix vr]
   (if (and (= (.mrows a) (.ncols a)) (< 0 (.mrows w))
            (or (nil? vl) (and (= (.mrows a) (.mrows vl)) (= (.mrows w) (.ncols vl))
                               (api/compatible? a vl) (api/fits-navigation? a vl)))
            (or (nil? vr) (and (= (.mrows a) (.mrows vr)) (= (.mrows w) (.ncols vr))
                               (api/compatible? a vr)(api/fits-navigation? a vr))))
     (api/ev (api/engine a) a w vl vr)
     (dragan-says-ex "You cannot compute eigenvalues of a non-square matrix or with the provided destinations."
                     {:a (info a) :w (info w) :vl (info vl) :vr (info vr) :errors
                      (cond-into []
                                 (not (= (.mrows a) (.ncols a))) "a is not a square matrix"
                                 (not (< 0 (.mrows w))) "w does not have at least one row"
                                 (and vl (not (= (.mrows w) (.ncols vl)))) "mrows of w and ncols of vl do not match"
                                 (and vr (not (= (.mrows w) (.ncols vr)))) "mrows of w and ncols of vl do not match"
                                 (and vl (not (= (.mrows a) (.mrows vl)))) "a and vl have different row dimensions"
                                 (and vr (not (= (.mrows a) (.mrows vr)))) "a and vr have different row dimensions"
                                 (and vl (not (api/fits-navigation? a vl))) "a and vl do not have the same layout"
                                 (and vr (not (api/fits-navigation? a vr))) "a and vr do not have the same layout")})))
  ([a w]
   (ev! a w nil nil))
  ([^Matrix a]
   (let-release [w (api/create-ge (api/factory a) (.mrows a) (if (symmetric? a) 1 2) true false)]
     (ev! a w))))

(defn ev
  "Computes eigenvalues without destroying `a`'s data."
  ([a]
   (with-release [a-copy (copy a)]
     (ev! a-copy))))

(defn es!
  "Computes the eigenvalues and Schur factorization of a matrix `a`.

  On exit, `a` is overwritten with the Schur form T. The first 2 columns of a column-oriented GE matrix
  `w` are overwritten with eigenvalues of `a`. If `v` GE matrice is provided, it will be overwritten
  by the orthogonal matrix Z of Schur vectors. If `vs` is nil, only eigenvalues are computed.

  If the QR algorithm failed to compute all the eigenvalues, throws ExceptionInfo, with the information
  on the index of the first eigenvalue that converged.
  If `w` is not column-oriented, or does not have 2 columns, throws ExceptionInfo.
  If `v`'s dimensions do not fit with `a`'s dimensions, throws ExceptionInfo.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info about [lapacke_?gees](https://software.intel.com/en-us/mkl-developer-reference-c-gees).
  "
  ([^Matrix a ^Matrix w ^Matrix vs]
   (if (and (= (.mrows a) (.ncols a))
            (= (.mrows a) (.mrows w)) (= 2 (.ncols w))
            (or (nil? vs) (and (= (.mrows a) (.mrows vs) (.ncols vs))
                               (api/compatible? a vs) (api/fits-navigation? a vs))))
     (api/es (api/engine a) a w vs)
     (dragan-says-ex "You cannot compute eigenvalues of a non-square matrix or with the provided destinations."
                     {:a (info a) :w (info w) :vs (info vs) :errors
                      (cond-into []
                                 (not (= (.mrows a) (.ncols a))) "a is not a square matrix"
                                 (not (= (.mrows a) (.mrows w))) "a and w have different row dimensions"
                                 (not (< 1 (.ncols w))) "w does not have 2 columns"
                                 (and vs (not (= (.mrows a) (.mrows vs) (.ncols vs))))
                                 "a and vs dimensions do not fit"
                                 (and vs (not (api/fits-navigation? a vs)))
                                 "a and vs do not have the same layout")})))
  ([^Matrix a w]
   (es! a w nil))
  ([^Matrix a]
   (let-release [w (api/create-ge (api/factory a) (.mrows a) 2 true false)]
     (es! a w nil))))

(defn es
  "Computes eigenvalues using Schur factorization without destroying `a`'s data."
  ([a]
   (with-release [a-copy (copy a)]
     (es! a-copy))))

;; ================= Singular Value Decomposition ================================================

(defn svd!
  "Computes the singular value decomposition of a matrix `a`.

  If `superb` is not provided, uses faster but slightly less precise divide and conquer method (SDD).

  On exit, `a`'s contents is destroyed, or, if `u` or `vt` are `nil`, overwritten with U or transposed V
  singular vectors of `a`. `s` is a diagonal matrix populated with sorted singular values.
  If the factorization does not converge, a diagonal matrix `superb` is populated with
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
              (= min-mn (.mrows sigma) (.mrows superb) (.ncols sigma) (.ncols superb)))
       (api/svd (api/engine a) a sigma u vt superb)
       (throw (ex-info "You can not do a singular value decomposition with incompatible or ill-fitting arguments."
                       {:a (info a) :s (info sigma) :u (info u) :vt (info vt) :superb (info superb)
                        :errors
                        (cond-into []
                                   (not (= min-mn (.mrows sigma))) "mrows of sigma is not (min m n)"
                                   (not (= min-mn (.ncols sigma))) "ncols of sigma is not (min m n)"
                                   (not (= min-mn (.mrows superb))) "mrows of superb is not (min m n)"
                                   (not (= min-mn (.ncols superb))) "ncols of superb is not (min m n)"
                                   (not (or (nil? u) (= m (.mrows u) (.ncols u)))) "u is not a mxm matrix"
                                   (not (or (nil? u) (= min-mn (.ncols u)))) "ncols of u is not equals (min m n)"
                                   (not (or (nil? u) (= n (.mrows u)))) "mrows of vt is not equals n"
                                   (not (or (nil? u) (api/fits-navigation? a u))) "a and u do not have the same layout"
                                   (not (or (nil? vt) (api/fits-navigation? a vt))) "a and vt do not have the same layout"
                                   (not (or (nil? vt) (= n (.mrows vt) (.ncols vt)))) "vt is not a nxn matrix"
                                   (not (or (nil? vt) (= min-mn (.mrows vt)))) "mrows of vt is not equals (min m n)"
                                   (not (or (nil? vt) (= n (.ncols vt)))) "ncols of vt is not equals n")})))))
  ([^Matrix a ^Matrix sigma ^Matrix u ^Matrix vt]
   (let [m (.mrows a)
         n (.ncols a)
         min-mn (min m n)]
     (if (and (or (nil? u) (and (or (= m (.mrows u) (.ncols u))
                                    (and (= m (.mrows u)) (= min-mn (.ncols u))))
                                (api/compatible? a u) (api/fits-navigation? a u)))
              (or (nil? vt) (and (or (= n (.mrows vt) (.ncols vt))
                                     (and (= min-mn (.mrows vt)) (= n (.ncols vt))))
                                 (api/compatible? a vt) (api/fits-navigation? a vt)))
              (= min-mn (.mrows sigma) (.ncols sigma)))
       (api/sdd (api/engine a) a sigma u vt)
       (throw (ex-info "You can not do a singular value decomposition with incompatible or ill-fitting arguments."
                       {:a (info a) :s (info sigma) :u (info u) :vt (info vt)
                        :errors
                        (cond-into []
                                   (not (= min-mn (.mrows sigma))) "mrows of sigma is not (min m n)"
                                   (not (= min-mn (.ncols sigma))) "ncols of sigma is not (min m n)"
                                   (not (or (nil? u) (= m (.mrows u) (.ncols u)))) "u is not a mxm matrix"
                                   (not (or (nil? u) (= min-mn (.ncols u)))) "ncols of u is not equals (min m n)"
                                   (not (or (nil? u) (= n (.mrows u)))) "mrows of vt is not equals n"
                                   (not (or (nil? u) (api/fits-navigation? a u))) "a and u do not have the same layout"
                                   (not (or (nil? vt) (api/fits-navigation? a vt))) "a and vt do not have the same layout"
                                   (not (or (nil? vt) (= n (.mrows vt) (.ncols vt)))) "vt is not a nxn matrix"
                                   (not (or (nil? vt) (= min-mn (.mrows vt)))) "mrows of vt is not equals (min m n)"
                                   (not (or (nil? vt) (= n (.ncols vt)))) "ncols of vt is not equals n")})))))
  ([^Matrix a ^Matrix sigma ^Matrix superb]
   (let [min-mn (min (.mrows a) (.ncols a))]
     (if (and (= min-mn (.mrows sigma) (.ncols sigma) (.mrows superb) (.ncols superb))
              (api/compatible? a sigma) (api/compatible? a superb))
       (api/svd (api/engine a) a sigma superb)
       (throw (ex-info "You can not do a singular value decomposition with incompatible or ill-fitting arguments."
                       {:a (info a) :s (info sigma) :superb (info superb) :errors
                        (cond-into []
                                   (not (= min-mn (.mrows sigma))) "mrows of sigma is not m"
                                   (not (= min-mn (.ncols sigma))) "ncols of sigma is not n"
                                   (not (= min-mn (.mrows superb))) "mrows of superb is not m"
                                   (not (= min-mn (.ncols superb))) "ncols of superb is not n")})))))
  ([^Matrix a ^Matrix sigma]
   (let [min-mn (min (.mrows a) (.ncols a))]
     (if (and (= min-mn (.mrows sigma) (.ncols sigma))
              (api/compatible? a sigma))
       (api/sdd (api/engine a) a sigma)
       (throw (ex-info "You can not do a singular value decomposition with incompatible or ill-fitting arguments."
                       {:a (info a) :s (info sigma) :errors
                        (cond-into []
                                   (not (= min-mn (.mrows sigma))) "mrows of sigma is not m"
                                   (not (= min-mn (.ncols sigma))) "ncols of sigma is not n")}))))))

(defn svd
  "Computes the singular value decomposition of a matrix `a`, and returns a SVD record containing
  `:sigma` (the singular values), `u` (if `u?` is true) and `vt` (if `vt?` is true).

  If `sdd?` is true, uses the faster divide and conquer SVD method unless. If `sdd?` is not provided,
  uses ordinary SVD if both `u?` and `vt?` are false.

  If the reduction to bidiagonal form failed to converge, throws ExceptionInfo, with the information
  on the number of converged superdiagonals.
  If some value in the native call is illegal, throws ExceptionInfo.

  See related info about [[svd!]]
  "
  ([^Matrix a ^Boolean u? ^Boolean vt? sdd?]
   (let [fact (api/factory a)
         min-mn (min (.mrows a) (.ncols a))]
     (let-release [u (if u? (ge fact (.mrows a) min-mn) nil)
                   vt (if vt? (ge fact min-mn (.ncols a)) nil)
                   sigma (gd fact min-mn)]
       (with-release [a-copy (copy a)]
         (if sdd?
           (api/sdd (api/engine a-copy) a-copy sigma u vt)
           (with-release [superb (raw sigma)]
             (api/svd (api/engine a-copy) a-copy sigma u vt superb)))
         (->SVDecomposition sigma u vt true)))))
  ([a u? vt?]
   (svd a u? vt? (or u? vt?)))
  ([a sdd?]
   (svd a false false sdd?))
  ([a]
   (svd a false false false)))
