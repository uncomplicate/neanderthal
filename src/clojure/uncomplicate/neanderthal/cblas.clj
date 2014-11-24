(ns uncomplicate.neanderthal.cblas
  (:require [primitive-math]
            [vertigo
             [bytes :refer [direct-buffer byte-seq
                            slice-buffer cross-section]]
             [structs :refer [float64 wrap-byte-seq]]]
            [uncomplicate.neanderthal.protocols :refer :all])
  (:import [uncomplicate.neanderthal CBLAS]
           [java.nio ByteBuffer]
           [uncomplicate.neanderthal.protocols 
            Block DoubleVector DoubleMatrix]))

(set! *warn-on-reflection* true)
(primitive-math/use-primitive-operators)

;-------------- Double Vector -----------------------------
(deftype DoubleBlockVector [^ByteBuffer arr ^long n ^long stride]
  Object
  (hashCode [this]
    (loop [i 0 res (hash-combine (hash :DoubleBlockVector) n)]
      (if (< i n)
        (recur (inc i) (hash-combine res (.entry this i)))
        res)))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (instance? DoubleBlockVector y)
     (and (= n (.dim ^DoubleBlockVector y))
          (loop [i 0]
            (if (< i n)
              (if (= (.entry x i) (.entry ^DoubleBlockVector y i))
                (recur (inc i))
                false)
              true)))
     :default false))
  clojure.lang.Seqable
  (seq [_]
    (wrap-byte-seq float64 (* 8 stride) 0 (byte-seq arr)))
  Carrier
  (pure [_]
    (DoubleBlockVector. (direct-buffer (* 8 n)) n 1))
  Block
  (buf [_]
    arr)
  (stride [_]
    stride)
  (length [_]
    n)
  DoubleVector
  (dim [_]
    n)
  (entry [_ i]
    (.getDouble arr (* 8 stride i)))
  (dot [_ y]
    (let [by ^Block y]
      (CBLAS/ddot n
                  arr stride 
                  (.buf by) (.stride by))))
  (nrm2 [_]
    (CBLAS/dnrm2 n arr stride))
  (asum [_]
    (CBLAS/dasum n arr stride))
  (iamax [_]
    (CBLAS/idamax n arr stride))
  (rot [x y c s]
    (let [by ^Block y]
      (do (CBLAS/drot n
                      arr stride 
                      (.buf by) (.stride by) 
                      c s)
          x)))
  (swap [x y]
    (let [by ^Block y]
      (do (CBLAS/dswap n
                       arr stride
                       (.buf by) (.stride by))
          x)))
  (scal [x alpha]
    (do (CBLAS/dscal n alpha arr stride)
        x))
  (copy [_ y]
    (let [by ^Block y]
      (do (CBLAS/dcopy n
                       arr stride
                       (.buf by) (.stride by))
          y)))
  (axpy [_ alpha y]
    (let [by ^Block y]
      (do (CBLAS/daxpy n alpha arr stride 
                       (.buf by) 
                       (.stride by))
          y))))

(defmethod print-method DoubleBlockVector
  [^DoubleBlockVector dv ^java.io.Writer w] 
  (.write w (format "#<DoubleBlockVector| n:%d, stride:%d %s>"
                    (.dim dv) (.stride dv) (pr-str (seq dv)))))

;; ================= GE General Matrix =====================
;; TODO all algorithms are for order=COLUMN_MAJOR and ld=m
(deftype DoubleGeneralMatrix [^ByteBuffer arr ^long m 
                              ^long n ^long ld ^long order]
  Object
  (hashCode [this]
    (let [dim (* m n)]
      (loop [i 0 res (hash-combine
                      (hash :DoubleGeneralMatrix) dim)]
        (if (< i dim)
          (recur (inc i)
                 (hash-combine res (.entry this (rem i m) (rem i n))))
          res))))
  (equals [x y]
    (cond
     (nil? y) false
     (identical? x y) true
     (instance? DoubleGeneralMatrix y)
     (let [dgy ^DoubleGeneralMatrix y
           arry (.buf dgy)
           lx (.length x)
           ly (.length dgy)
           ldy (.stride dgy)]
       (and (= (.mrows x) (.mrows dgy))
            (= (.ncols x) (.ncols dgy))
            (loop [i 0]
              (if (< i lx)
                (if (= (.getDouble arr (* 8 i))
                       (.getDouble ^ByteBuffer arry (* 8 i)))
                  (recur (inc i))
                  false)
                true))))
     :default false))
  Carrier
  (pure [_]
    (DoubleGeneralMatrix. (direct-buffer (* 8 m n)) m n m order))
  clojure.lang.Seqable
  (seq [_]
    (map (partial wrap-byte-seq float64)
         (cross-section (byte-seq arr) 0 (* 8 m) (* 8 m))))
  Block
  (buf [_]
    arr)
  (stride [_]
    ld)
  (length [_]
    (/ (.capacity ^ByteBuffer arr) 8))
  DoubleMatrix
  (mrows [_]
    m)
  (ncols [_]
    n)
  (entry [_ i j]
    (.getDouble arr (+ (* 8 m j) (* 8 i))))
  (row [a i]
    (DoubleBlockVector.
     (slice-buffer arr (* 8 i) (* 8 (- (.length a) i)))
     n m))
  (col [_ i]
    (DoubleBlockVector.
     (slice-buffer arr (* 8 m i) (* 8 m))
     m 1))
  (mv [_ alpha x beta y transa]
    (let [bx ^Block x
          by ^Block y]
      (do (CBLAS/dgemv order transa
                       m n
                       alpha
                       arr ld
                       (.buf bx) (.stride bx)
                       beta
                       (.buf by) (.stride by))
          y)))
  #_(rank [a alpha x y] ;;TODO
    (let [bx ^Block x
          by ^Block y]
      (do (CBLAS/dger order
                      m n
                      alpha
                      (.buf bx) (.stride bx)
                      (.buf by) (.stride by)
                      arr ld)
          a)))
  (mm [_ alpha b beta c transa transb]
    (let [bb ^Block b
          bc ^Block c]
      (do (CBLAS/dgemm order
                       transa transb
                       m (.ncols b) n
                       alpha
                       arr ld
                       (.buf bb) (.stride bb)
                       beta
                       (.buf bc) (.stride bc))
          c))))

(defmethod print-method DoubleGeneralMatrix
  [^DoubleGeneralMatrix m ^java.io.Writer w] 
  (.write w (format "#<DoubleGeneralMatrix| m:%d, n:%d ld:%d %s>"
                    (.mrows m) (.ncols m) (.stride m) (pr-str (seq m)))))

(primitive-math/unuse-primitive-operators)
