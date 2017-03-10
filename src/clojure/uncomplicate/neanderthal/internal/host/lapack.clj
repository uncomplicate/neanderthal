;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.host.lapack
  (:import [uncomplicate.neanderthal.internal.host CBLAS]))

(defmacro with-lapack-check [expr]
  ` (let [err# ~expr]
      (if (zero? err#)
        err#
        (throw (IllegalArgumentException. (format "LAPACK error: %d" err#))))))

;; =========================== Auxiliary LAPACK Routines =========================

;; ----------------- Common vector matrix macros and functions -----------------------

(defmacro vctr-laset [method alpha x]
  `(with-lapack-check
     (~method (int (if (= 1 (.stride ~x)) CBLAS/ORDER_COLUMN_MAJOR CBLAS/ORDER_ROW_MAJOR))
      (int \g) (.dim ~x) 1 ~alpha ~alpha (.buffer ~x) (.offset ~x) (.stride ~x))))

;; ----------------- Common GE matrix macros and functions -----------------------

(defmacro ge-lan [method norm a]
  `(~method (.order ~a) ~norm (.mrows ~a) (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a)))

(defmacro ge-laset [method alpha beta a]
  `(with-lapack-check
     (~method (.order ~a) (int \g) (.mrows ~a) (.ncols ~a)
      ~alpha ~beta (.buffer ~a) (.offset ~a) (.stride ~a))))

;; ----------------- Common TR matrix macros and functions -----------------------

;; There seems to be a bug in MKL's LAPACK_?lantr. If the order is column major,
;; it returns 0.0 as a result. To fix this, I had to do the uplo# trick.
(defmacro tr-lan [method norm a]
  `(let [uplo# (if (= CBLAS/ORDER_COLUMN_MAJOR (.order ~a))
                 (if (= CBLAS/UPLO_LOWER (.uplo ~a)) \L \U)
                 (if (= CBLAS/UPLO_LOWER (.uplo ~a)) \U \L))]
     (~method CBLAS/ORDER_ROW_MAJOR ~norm
      (int uplo#) (int (if (= (CBLAS/DIAG_UNIT) (.diag ~a)) \U \N))
      (.mrows ~a) (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a))))

(defmacro tr-lacpy [stripe-nav lacpy copy a b]
  `(if (= (.order ~a) (.order ~b))
     (with-lapack-check
       (~lacpy (.order ~a) (int (if (= (CBLAS/UPLO_UPPER) (.uplo ~a)) \U \L)) (.mrows ~a) (.ncols ~a)
        (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~b) (.offset ~b) (.stride ~b)))
     (let [n# (.fd ~a)
           ld-a# (.stride ~a)
           offset-a# (.offset ~a)
           buff-a# (.buffer ~a)
           ld-b# (.stride ~b)
           offset-b# (.offset ~b)
           buff-b# (.buffer ~b)]
       (dotimes [j# n#]
         (let [start# (.start ~stripe-nav n# j#)
               n-j# (- (.end ~stripe-nav n# j#) start#)]
           (~copy n-j# buff-a# (+ offset-a# (* ld-a# j#) start#) 1
            buff-b# (+ offset-b# j# (* ld-b# start#)) n#))))))

(defmacro tr-lascl [method alpha a]
  `(with-lapack-check
     (~method (.order ~a) (int (if (= (CBLAS/UPLO_UPPER) (.uplo ~a)) \U \L))
      0 0 1.0 ~alpha (.mrows ~a) (.ncols ~a) (.buffer ~a) (.offset ~a) (.stride ~a))))

(defmacro tr-laset [method alpha beta a]
  `(with-lapack-check
     (~method (.order ~a) (int (if (= (CBLAS/UPLO_UPPER) (.uplo ~a)) \U \L))
      (.mrows ~a) (.ncols ~a) ~alpha ~beta (.buffer ~a) (.offset ~a) (.stride ~a))))

;; =========== Drivers and Computational LAPACK Routines ===========================

;; ------------- Singular Value Decomposition LAPACK -------------------------------

(defmacro with-sv-check [ipiv expr]
  `(if (= 1 (.stride ~ipiv))
     (let [info# ~expr]
       (cond
         (= 0 info#) ~ipiv
         (< 0 info#) (throw (IllegalArgumentException. "TODO Illegal i"))
         :else (throw (RuntimeException. "TODO Singular, no solution"))))
     (throw (IllegalArgumentException. "TODO Illegal ipiv stride."))))

(defmacro ge-trf [method a ipiv]
  `(with-sv-check ~ipiv
     (~method (.order ~a) (.mrows ~a) (.ncols ~a)
      (.buffer ~a) (.offset ~a) (.stride ~a) (.buffer ~ipiv) (.offset ~ipiv))))

(defmacro ge-trs [method a b ipiv]
  `(with-sv-check ~ipiv
     (~method (.order ~a) (int (if (= (.order ~a) (.order ~b)) \N \T))
      (.mrows ~b) (.ncols ~b) (.buffer ~a) (.offset ~a) (.stride ~a)
      (.buffer ~ipiv) (.offset ~ipiv) (.buffer ~b) (.offset ~b) (.stride ~b))))

(defmacro ge-sv [method a b ipiv]
  `(with-sv-check ~ipiv
     (~method (.order ~a) (.mrows ~b) (.ncols ~b) (.buffer ~a) (.offset ~a) (.stride ~a)
      (.buffer ~ipiv) (.offset ~ipiv) (.buffer ~b) (.offset ~b) (.stride ~b))))
