(ns uncomplicate.neanderthal.impl.matrix-api
  "Namespace for the core.matrix implementation of Neanderthal.

   Should not be used directly by user code. Users should instead create and manipulate Neanderthal
   vectors and matrices using the core.matrix API"
  (:require [uncomplicate.commons.core :refer [release let-release]]
            [uncomplicate.neanderthal.core :as core] 
            [clojure.core.matrix :refer :all]
            [clojure.core.matrix.protocols :as mp]
            [uncomplicate.neanderthal.impl.cblas :as cblas]
            [clojure.core.matrix.implementations :as imp]
            [uncomplicate.neanderthal
             [protocols :as p]])
  (:import [uncomplicate.neanderthal.protocols Vector Matrix Block
            BLAS BLASPlus Changeable RealChangeable DataAccessor]))

(set! *warn-on-reflection* true)
(set! *unchecked-math* :warn-on-boxed)

(defmacro neanderthal?
  "Returns true if v is a Neanderthal class (i.e. an instance of Matrix or Vector)"
  ([a]
    `(or (instance? Vector ~a) (instance? Matrix ~a))))

(defmacro double-coerce 
  "Coerces a 0-dimensional object to a double value"
  ([x]
  `(let [x# ~x]
     (double (if (number? x#) x# (mp/get-0d x#))))))

(defmacro error
  "Throws an error with the provided message(s). This is a macro in order to try and ensure the 
   stack trace reports the error at the correct source line number."
  ([& messages]
    `(throw (Exception. (str ~@messages)))))

(eval
  `(extend-protocol mp/PImplementation
     ~@(mapcat 
         (fn [sym]
           (cons sym
             '(
                (implementation-key [m] :neanderthal)
                (supports-dimensionality? [m dims] (<= 1 (long dims) 2))
                (new-vector [m length] (core/create cblas/cblas-double (long length)))
                (new-matrix [m rows columns] (core/create cblas/cblas-double (long rows) (long columns)))
                (new-matrix-nd [m shape] 
                               (case (count shape)
                                 0 0.0
                                 1 (core/create cblas/cblas-double (long (first shape)))
                                 2 (core/create cblas/cblas-double (long (first shape)) (long (second shape)))
                                 (let [d0 (first shape)
                                       moredims (next shape)]
                                   (mapv 
                                     (fn [ds] (mp/new-matrix-nd m ds))
                                     (range d0)))))
                (construct-matrix [m data]
                                  (let [dims (long (mp/dimensionality data))]
                                    (cond 
                                      (neanderthal? data) (core/copy data)
                                      (mp/is-scalar? data) (double-coerce data)
                                      (== 1 dims) (core/create-vector cblas/cblas-double (to-nested-vectors data))
                                      (== 2 dims) (let [shp (shape data)
                                                        rows (long (first shp))
                                                        cols (long (second shp))]
                                                    (core/create-ge-matrix cblas/cblas-double
                                                        rows cols 
                                                        ;; note we have to transpose since Neanderthal expects
                                                        ;; elements in column-major order
                                                        (mp/element-seq (transpose data)))) 
                                    :default
                                      (let [vm (mp/construct-matrix [] data)] 
                                        ;; (println m vm (shape vm))
                                         (assign! (mp/new-matrix-nd m (shape vm)) vm))))))))
         '[uncomplicate.neanderthal.protocols.Vector uncomplicate.neanderthal.protocols.Matrix]) ))

(extend-protocol mp/PDimensionInfo
  Vector
    (dimensionality [m]
      1)
    (is-vector? [m]
      true)
    (is-scalar? [m]
      false)
    (get-shape [m]
      [(long (.dim m))])
    (dimension-count [m x]
      (if (== 0 (long x))
        (long (.dim m))
        (error "Vector does not have dimension: " x)))
  Matrix
    (dimensionality [m]
      2)
    (is-vector? [m]
      false)
    (is-scalar? [m]
      false)
    (get-shape [m]
      [(long (.mrows m)) (long (.ncols m))])
    (dimension-count [m x]
      (let [x (long x)]
        (cond 
          (== x 0) (.mrows m)
          (== x 1) (.ncols m)
          :else (error "Matrix does not have dimension: " x)))))

(extend-protocol mp/PObjectArrayOutput
  Vector
	  (to-object-array [m]
	    (let [ec (.dim m)
	          ^objects obs (object-array ec)]
	      (dotimes [i ec] (aset obs i (.boxedEntry m i))) 
	      obs))
	  (as-object-array [m]
	    nil)
   Matrix
	  (to-object-array [m]
	    (let [rows (.mrows m)
	          cols (.ncols m)
            ^objects obs (object-array (* rows cols))]
	      (dotimes [i rows] 
          (dotimes [j cols] (aset obs (+ j (* cols i)) (.boxedEntry m i j)))) 
	      obs))
	  (as-object-array [m]
	    nil))

(extend-protocol mp/PIndexedAccess
  Vector
    (get-1d [m i]
      (.boxedEntry m (long i)))
    (get-2d [m i j]
      (error "Can't access 2-dimensional index of a vector"))
    (get-nd [m indexes]
      (when-not (== 1 (count indexes)) (error "Invalid index for Vector: " indexes))
      (.boxedEntry m (long (first indexes))))
  Matrix
    (get-1d [m i]
      (error "Can't access 1-dimensional index of a matrix"))
    (get-2d [m i j]
      (.boxedEntry m (long i) (long j)))
    (get-nd [m indexes]
      (when-not (== 2 (count indexes)) (error "Invalid index for Matrix: " indexes))
      (.boxedEntry m (long (first indexes)) (long (second indexes)))))

(extend-protocol mp/PIndexedSettingMutable
  Vector
    (set-1d! [m i v] (.set ^RealChangeable m (long i) (double-coerce v)))
    (set-2d! [m i j v] (error "Can't do 2-dimensional set on a 1D vector!"))
    (set-nd! [m indexes v]
      (if (== 1 (count indexes))
        (.set ^RealChangeable m (long (first indexes)) (double-coerce v))
        (error "Can't do " (count indexes) "-dimensional set on a 1D vector!"))) 
  Matrix
    (set-1d! [m i v] (error "Can't do 1-dimensional set on a 2D matrix!"))
    (set-2d! [m i j v] (.set ^RealChangeable m (long i) (long j) (double-coerce v)))
    (set-nd! [m indexes v]
      (if (== 2 (count indexes))
        (.set ^RealChangeable m (long (first indexes)) (long (second indexes)) (double-coerce v))
        (error "Can't do " (count indexes) "-dimensional set on a 2D matrix!"))))

(extend-protocol mp/PSubVector
  Vector
    (subvector [m start length]
      (let [k (long start)
            l (long length)]
        (.subvector m k l)))) 

(extend-protocol mp/PSliceView
  Matrix
    (get-major-slice-view [m i] 
      (.row m (long i))))

(extend-protocol mp/PSliceView2
  Matrix
    (get-slice-view [m dim i]
      (case (long dim) 
        0 (.row m (long i))
        1 (.col m (long i))
        (error "Can't slice on dimension " dim " of a Matrix"))))

(extend-protocol mp/PSliceSeq
  Matrix  
    (get-major-slice-seq [m] 
      (mapv (fn [i] (.row m (long i)))
            (range (.mrows m)))))

(extend-protocol mp/PTranspose
  Vector (transpose [m] m)
  Matrix (transpose [m] (.transpose m))) 

;; Register the Neanderthal implementation using CBLAS
(imp/register-implementation (core/create cblas/cblas-double 3))

