;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.common
  (:require [clojure.pprint :refer [cl-format]]
            [uncomplicate.fluokitten.core :refer [fold]]
            [uncomplicate.commons.core :refer [Releaseable release let-release double-fn]]
            [uncomplicate.neanderthal.math :refer [ceil floor]]
            [uncomplicate.neanderthal.internal.api :refer :all])
  (:import [clojure.lang Seqable IFn]
           [uncomplicate.neanderthal.internal.api Vector Matrix TRMatrix GEMatrix SYMatrix
            DataAccessor SegmentVector UploNavigator]))

(defn ^:private unsupported []
  (throw (UnsupportedOperationException. (format "This operation is not supported in wrappers."))))

(def ^:private compile-format #'clojure.pprint/compile-format)

(def format-g (compile-format "~6,2,,1G "))
(def format-f (compile-format "~7,2F "))
(def format-a (compile-format "~3@T~A~3@T"))

(defn format-vector [^java.io.Writer w formatter ^Vector x]
  (let [n-print (min (.dim x) (long (or *print-length* 16)))
        start-2 (- (.dim x) (floor (/ n-print 2)))]
    (dotimes [i (ceil (/ n-print 2))]
      (cl-format w formatter (.boxedEntry x i)))
    (when (< n-print (.dim x))
      (cl-format w format-a "..."))
    (dotimes [i (floor (/ n-print 2))]
      (cl-format w formatter (.boxedEntry x (+ start-2 i))))))

(defn format-ge [^java.io.Writer w formatter ^Matrix a max-value]
  (let [pl (long (or *print-length* 16))
        m-print (min (.mrows a) pl)
        n-print (min (.ncols a) pl)
        m-start-2 (- (.mrows a) (floor (/ m-print 2)))]
    (dotimes [i (ceil (/ m-print 2))]
      (.write w "\n")
      (format-vector w formatter (.row a i)))
    (when (< m-print (.mrows a))
      (let [width (* 0.1 n-print (.length ^String (cl-format nil formatter max-value)))]
        (dotimes [_ 3]
          (.write w "\n")
          (dotimes [_ width]
            (.write w "     .     ")))
        (.write w "\n")))
    (dotimes [i (floor (/ m-print 2))]
      (.write w "\n")
      (format-vector w formatter (.row a (+ m-start-2 i))))))

;; ============ Segment Vector ====================================================

(declare segment-vector)

(deftype WrappedSegmentVector [^Vector seg ^long n ^long fstart ^long unit-idx]
  Object
  (hashCode [x]
    (-> (hash :SegmentVector) (hash-combine n) (hash-combine unit-idx)
        (hash-combine (nrm2 (engine seg) seg))))
  (equals [x y]
    (cond
      (nil? y) false
      (identical? x y) true
      (and (instance? SegmentVector y) (compatible? x y) (fits? x y))
      (= seg (.segment ^SegmentVector y))
      :default false))
  (toString [_]
    (format "#SegmentVector[n:%d, segment:%s, %s]"
            n [fstart (+ fstart (.dim seg))] (if (= -1 unit-idx) "non-unit" (format "unit:%s" unit-idx))))
  Releaseable
  (release [_]
    (release seg))
  Seqable
  (seq [_]
    (seq seg))
  Container
  (raw [_]
    (segment-vector (raw seg) n fstart unit-idx))
  (raw [_ fact]
    (segment-vector (raw seg fact) n fstart unit-idx))
  (zero [_]
    (segment-vector (zero seg) n fstart unit-idx))
  (zero [_ fact]
    (segment-vector (zero seg fact) n fstart unit-idx))
  (host [x]
    (segment-vector (host seg) n fstart unit-idx))
  (native [x]
    (let-release [native-seg (native seg)]
      (if (identical? seg native-seg)
        x
        (segment-vector native-seg n fstart unit-idx))))
  MemoryContext
  (fully-packed? [_]
    (fully-packed? seg))
  (compatible? [_ y]
    (compatible? seg y))
  (fits? [_ y]
    (let [y ^SegmentVector y]
      (and (= n (.dim y)) (= fstart (.start y)) (= unit-idx (.unitIndex y)) (fits? seg (.segment y)))))
  EngineProvider
  (engine [x]
    x)
  FactoryProvider
  (factory [_]
    (factory seg))
  (native-factory [_]
    (native-factory seg))
  (index-factory [_]
    (index-factory seg))
  DataAccessorProvider
  (data-accessor [_]
    (data-accessor seg))
  SegmentVector
  (unitIndex [_]
    unit-idx)
  (start [_]
    fstart)
  (segment [_]
    seg)
  IFn
  (invoke [x i]
    (.boxedEntry x i))
  (invoke [x]
    n)
  Vector
  (dim [_]
    n)
  (boxedEntry [x i]
    (let [idx (- i fstart)]
      (if (< -1 idx (.dim seg))
        (.boxedEntry seg idx)
        (if (= unit-idx i)
          1.0
          0.0))))
  Blas
  (swap [_ x y]
    (swap (engine seg) (.segment ^SegmentVector x) (.segment ^SegmentVector y))
    x)
  (copy [_ x y]
    (copy (engine seg) (.segment ^SegmentVector x) (.segment ^SegmentVector y))
    y))

(defn segment-vector
  ([^Vector seg ^long n ^long start ^long unit-idx]
   (let [end (+ start (.dim seg))]
     (if (and (<= 0 start end n) (or (= -1 unit-idx) (= (dec start) unit-idx) (= end unit-idx)))
       (->WrappedSegmentVector seg n start unit-idx)
       (throw (ex-info "Segment not in scope." {:start start :end end :unit-index unit-idx})))))
  ([^Vector seg ^long n ^long start]
   (segment-vector seg n start -1)))

(defmethod print-method SegmentVector [^SegmentVector x ^java.io.Writer w]
  (if (< 0 (.dim x))
    (let [segment (.segment x)
          end (+ (.start x) (.dim segment))
          unit-idx (.unitIndex x)
          pre-unit (< -1 unit-idx (.start x))
          post-unit (< end (.unitIndex x))
          max-value (double (amax (engine segment) segment))
          min-value (.boxedEntry segment (iamin (engine segment) segment))
          formatter (if (and (not (< 0.0 min-value 0.01)) (< max-value 10000.0)) format-f format-g)]
      (.write w (str x "\n["))
      (if (and (not= 0 unit-idx) (< 0 (.start x))) (cl-format w format-a "***"))
      (if pre-unit (cl-format w format-a "* 1 *"))
      (format-vector w formatter segment)
      (if post-unit (cl-format w format-a "* 1 *"))
      (if (and (not= (.dim x) unit-idx) (< end (.dim x))) (cl-format w format-a "***"))
      (.write w "]\n"))
    (.write w (str x))))

(defn format-tr [^java.io.Writer w formatter ^UploNavigator uplo-nav ^TRMatrix a max-value]
  (let [pl (long (or *print-length* 16))
        m-print (min (.mrows a) pl)
        n-print (min (.ncols a) pl)
        m-start-2 (- (.mrows a) (floor (/ m-print 2)))]
    (dotimes [i (ceil (/ m-print 2))]
      (.write w "\n")
      (format-vector w formatter
                     (segment-vector (.row a i) (.ncols a)
                                     (.rowStart uplo-nav (.ncols a) i) (.unitIndex uplo-nav i))))
    (when (< m-print (.mrows a))
      (let [width (* 0.1 n-print (.length ^String (cl-format nil formatter max-value)))]
        (dotimes [_ 3]
          (.write w "\n")
          (dotimes [_ width]
            (.write w "     .     ")))
        (.write w "\n")))
    (dotimes [i (floor (/ m-print 2))]
      (let [i (+ m-start-2 i)]
        (.write w "\n")
        (format-vector w formatter
                       (segment-vector (.row a i) (.ncols a)
                                       (.rowStart uplo-nav (.ncols a) i) (.unitIndex uplo-nav i)))))))

(deftype SYRowColumn [^Vector seg1 ^Vector seg2]
  Vector
  (dim [_]
    (max 0 (dec (+ (.dim seg1) (.dim seg2)))))
  (boxedEntry [x i]
    (if (< -1 i (.dim seg1))
      (.boxedEntry seg1 i)
      (.boxedEntry seg2 (inc (- i (.dim seg1))) ))))

(defn format-sy [^java.io.Writer w formatter ^SYMatrix a max-value]
  (let [pl (long (or *print-length* 16))
        m-print (min (.mrows a) pl)
        n-print (min (.ncols a) pl)
        m-start-2 (- (.mrows a) (floor (/ m-print 2)))
        lower (= LOWER (.uplo a)) ]
    (dotimes [i (ceil (/ m-print 2))]
      (.write w "\n")
      (format-vector w formatter (if lower
                                   (->SYRowColumn (.row a i) (.col a i))
                                   (->SYRowColumn (.col a i) (.row a i)))))
    (when (< m-print (.mrows a))
      (let [width (* 0.1 n-print (.length ^String (cl-format nil formatter max-value)))]
        (dotimes [_ 3]
          (.write w "\n")
          (dotimes [_ width]
            (.write w "     .     ")))
        (.write w "\n")))
    (dotimes [i (floor (/ m-print 2))]
      (let [i (+ m-start-2 i)]
        (.write w "\n")
        (format-vector w formatter (if lower
                                     (->SYRowColumn (.col a i) (.row a i))
                                     (->SYRowColumn (.col a i) (.row a i))))))))

;; ======================== LU factorization ==========================================

(def ^:private f* (double-fn *))
(def ^:private falsify (constantly false))

(defn ^:private stale-factorization []
  (throw (ex-info "Cannot compute with stale LU factorization." {})))

(defrecord LUFactorization [^GEMatrix lu ^GEMatrix a ^Vector ipiv ^Boolean master fresh]
  Releaseable
  (release [_]
    (when master (release lu))
    (release ipiv))
  LU
  (lu-trs [_ b]
    (if @fresh
      (trs (engine lu) lu b ipiv)
      (stale-factorization)))
  (lu-tri! [_]
    (if (compare-and-set! fresh true false)
      (tri (engine lu) lu ipiv)
      (stale-factorization)))
  (lu-tri [_]
    (if @fresh
      (let-release [res (raw lu)]
        (let [eng (engine lu)]
          (tri eng (copy eng lu res) ipiv))
        res)
      (stale-factorization)))
  (lu-con [_ nrm nrm1?]
    (if @fresh
      (con (engine lu) lu nrm nrm1?)
      (stale-factorization)))
  (lu-con [_ nrm1?]
    (if a
      (if @fresh
        (con (engine lu) lu (if nrm1? (nrm1 (engine a) a) (nrmi (engine a) a)) nrm1?)
        (stale-factorization))
      (throw (ex-info "Cannot estimate the condition number without the reference to the original GE matrix." {}))))
  (lu-det [_]
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

(defn create-lu
  ([lu a ipiv]
   (->LUFactorization lu a ipiv true (atom true)))
  ([lu ipiv]
   (->LUFactorization lu nil ipiv false (atom true))))
