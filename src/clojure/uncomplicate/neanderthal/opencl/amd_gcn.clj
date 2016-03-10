(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.amd-gcn
  (:refer-clojure :exclude [accessor])
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl
             [core :refer :all]
             [toolbox :refer [count-work-groups enq-reduce enq-reduce-horizontal
                              enq-read-int enq-read-double wrap-int]]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [math :refer [sqrt]]]
            [uncomplicate.neanderthal.opencl.clblock :refer :all])
  (:import [uncomplicate.clojurecl.core WorkSize CLBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Block Matrix DataAccessor]))

(deftype GCNVectorEngine [^long WGS
                          claccessor queue cl-buf
                          ^long n ^long ofst ^long strd
                          eq-flag
                          reduce-acc
                          reduce-iacc
                          linear-work-size
                          equals-vector-kernel
                          swap-kernel
                          copy-kernel
                          scal-kernel
                          axpy-kernel
                          sum-reduction-kernel
                          imax-reduction-kernel
                          dot-reduce-kernel
                          nrm2-reduce-kernel
                          asum-reduce-kernel
                          iamax-reduce-kernel
                          sum-reduce-kernel
                          imax-reduce-kernel
                          imin-reduce-kernel]

  Releaseable
  (release [_]
    (and
     (release eq-flag)
     (release reduce-acc)
     (release reduce-iacc)
     (release equals-vector-kernel)
     (release swap-kernel)
     (release copy-kernel)
     (release scal-kernel)
     (release axpy-kernel)
     (release sum-reduction-kernel)
     (release imax-reduction-kernel)
     (release dot-reduce-kernel)
     (release nrm2-reduce-kernel)
     (release asum-reduce-kernel)
     (release iamax-reduce-kernel)
     (release sum-reduce-kernel)
     (release imax-reduce-kernel)
     (release imin-reduce-kernel)))
  BlockEngine
  (equals-block [_ _ y]
    (if (< 0 n)
      (let [res (wrap-int 0)]
        (set-args! equals-vector-kernel 4 (.buffer ^Block y)
                   (wrap-int (.offset ^Block y)) (wrap-int (.stride ^Block y)))
        (enq-fill! queue eq-flag res)
        (set-arg! equals-vector-kernel 0 eq-flag)
        (enq-nd! queue equals-vector-kernel linear-work-size)
        (enq-read! queue eq-flag res)
        (= 0 (aget res 0)))
      true))
  BLAS
  (swap [_ _ y]
    (if (< 0 n)
      (do
        (set-arg! swap-kernel 1 (wrap-int ofst))
        (set-args! swap-kernel 3 (.buffer y) (wrap-int (.offset y))
                   (wrap-int (.stride y)))
        (enq-nd! queue swap-kernel linear-work-size))
      queue))
  (copy [_ x y]
    (if (< 0 n)
      (do
        (set-arg! copy-kernel 1 (wrap-int ofst))
        (set-args! copy-kernel 3 (.buffer y) (wrap-int (.offset y))
                   (wrap-int (.stride y)))
        (enq-nd! queue copy-kernel linear-work-size))
      queue))
  (dot [_ _ y]
    (if (< 0 n)
      (do
        (set-args! dot-reduce-kernel 4 (.buffer y) (wrap-int (.offset y))
                   (wrap-int (.stride y)))
        (enq-reduce queue dot-reduce-kernel sum-reduction-kernel WGS n)
        (enq-read-double queue reduce-acc))
      0.0))
  (nrm2 [_ _]
    (if (< 0 n)
      (do
        (enq-reduce queue nrm2-reduce-kernel sum-reduction-kernel WGS n)
        (sqrt (enq-read-double queue reduce-acc)))
      0.0))
  (asum [_ _]
    (if (< 0 n)
      (do
        (enq-reduce queue asum-reduce-kernel sum-reduction-kernel WGS n)
        (enq-read-double queue reduce-acc))
      0.0))
  (iamax [_ _]
    (if (< 0 n)
      (do
        (enq-reduce queue iamax-reduce-kernel imax-reduction-kernel WGS n)
        (enq-read-int queue reduce-iacc))
      0))
  (rot [_ _ y c s]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (rotg [_ _]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (rotm [_ _ y p]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (rotmg [_ _ args]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (scal [_ alpha _];;TODO zero-length, (< 0 n)
    (if (< 0 n)
      (do
        (set-args! scal-kernel 0 (wrap-prim claccessor alpha))
        (enq-nd! queue scal-kernel linear-work-size))
      queue))
  (axpy [_ alpha x y]
    (if (< 0 n)
      (do
        (set-args! axpy-kernel 0 (wrap-prim claccessor alpha))
        (set-args! axpy-kernel 4 (.buffer y) (wrap-int (.offset y))
                   (wrap-int (.stride y)))
        (enq-nd! queue axpy-kernel linear-work-size))
      queue))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (if (< 0 n)
      (do
        (set-arg copy-kernel 1 (wrap-int (+ ofst (long kx))))
        (set-args! copy-kernel 3 (.buffer y) (wrap-int (+ (.offset y) (long ky)))
                   (wrap-int (.stride y)))
        (enq-nd! queue copy-kernel (work-size-1d lx)))
      queue))
  (sum [_ x]
    (if (< 0 n)
      (do
        (enq-reduce queue sum-reduce-kernel sum-reduction-kernel WGS n)
        (enq-read-double queue reduce-acc))
      0.0))
  (imax [_ x]
    (if (< 0 n)
      (do
        (enq-reduce queue imax-reduce-kernel imax-reduction-kernel WGS n)
        (enq-read-int queue reduce-iacc))
      0))
  (imin [_ x]
    (if (< 0 n)
      (do
        (enq-reduce queue imin-reduce-kernel imax-reduction-kernel WGS n)
        (enq-read-int queue reduce-iacc))
      0)))

;; ======================= Dense Matrix ========================================

(deftype GCNMatrixEngine [^long WGSn ^long TS ^long WPT
                          claccessor queue
                          ^long m ^long n ^long ofst ^long ld ^long ord
                          eq-flag
                          reduce-acc
                          linear-work-size
                          equals-matrix-kernel
                          axpby-kernel
                          sum-reduction-horizontal-kernel
                          copy-matrix-kernel
                          swap-matrix-kernel
                          gerk-kernel
                          gemv-reduce-kernel
                          gemm-tiled-kernel
                          gemm-tiled-fit-kernel]
  Releaseable
  (release [_]
    (and
     (release eq-flag)
     (release reduce-acc)
     (release equals-matrix-kernel)
     (release axpby-kernel)
     (release sum-reduction-horizontal-kernel)
     (release swap-matrix-kernel)
     (release copy-matrix-kernel)
     (release gerk-kernel)
     (release gemv-reduce-kernel)
     (release gemm-tiled-kernel)
     (release gemm-tiled-fit-kernel)))
  BlockEngine
  (equals-block [_ _ b]
    (let [res (wrap-int 0)]
      (set-args! equals-matrix-kernel 4 (.buffer ^Block b)
                 (wrap-int (.offset ^Block b)) (wrap-int (.stride ^Block b)))
      (enq-fill! queue eq-flag res)
      (set-arg! equals-matrix-kernel 0 eq-flag)
      (enq-nd! queue equals-matrix-kernel (work-size-2d m n))
      (enq-read! queue eq-flag res)
      (= 0 (aget res 0))))
  BLAS
  (copy [_ a b]
    (if (< 0 (* m n))
      (do
        (set-args! copy-matrix-kernel 3 (.buffer b) (wrap-int (.offset b))
                   (wrap-int (.stride b)))
        (enq-nd! queue copy-matrix-kernel (work-size-2d m n)))
      queue))
  (swap [_ a b]
    (if (< 0 (* m n))
      (do
        (set-args! swap-matrix-kernel 3 (.buffer b) (wrap-int (.offset b))
                   (wrap-int (.stride b)))
        (enq-nd! queue swap-matrix-kernel (work-size-2d m n)))
      queue))
  (rank [_ alpha x y a]
    (if (< 0 (* m n))
      (do
        (set-args! gerk-kernel 0 (wrap-prim claccessor alpha)
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-nd! queue gerk-kernel (work-size-2d m n)))
      queue))
  (mv [_ alpha _ x beta y]
    (do
      (set-arg! gemv-reduce-kernel 1 (wrap-prim claccessor alpha))
      (set-args! gemv-reduce-kernel 5 (.buffer x) (wrap-int (.offset x))
                 (wrap-int (.stride x)))
      (enq-reduce-horizontal queue gemv-reduce-kernel
                             sum-reduction-horizontal-kernel
                             WGSn m n)
      (set-args! axpby-kernel 4 (wrap-prim claccessor beta) (.buffer y)
                 (wrap-int (.offset y)) (wrap-int (.stride y)))
      (enq-nd! queue axpby-kernel linear-work-size)))
  (mm [_ alpha a b beta c]
    (let [bn (.ncols ^Matrix b)
          RTS (long( / TS WPT))
          gemm-kernel (if (= 0 (mod m TS) (mod bn TS))
                        gemm-tiled-fit-kernel
                        gemm-tiled-kernel)]
      (set-arg! gemm-kernel 0 (wrap-prim claccessor alpha))
      (set-args! gemm-kernel 4
                 (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                 (wrap-prim claccessor beta)
                 (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c))
                 (wrap-int m) (wrap-int n) (wrap-int bn))
      (enq-nd! queue gemm-kernel
               (work-size-2d (* TS (count-work-groups TS m))
                             (* RTS (count-work-groups TS bn)))))))

(deftype GCNFactory [^DataAccessor claccessor ctx queue prog
                     ^long WGS WGSn TS WPT]
  Releaseable
  (release [_]
    (release prog))
  FactoryProvider
  (factory [_]
    (factory claccessor))
  Factory
  (create-vector [this n buf _]
    (if (and (<= 0 (long n) (.count claccessor buf)) (instance? CLBuffer buf))
      (->CLBlockVector this claccessor (vector-engine this [buf n 0 1])
                       (.entryType claccessor) true buf n 0 1)
      (throw (IllegalArgumentException.
              (format "I can not create an %d element vector from %d-element %s."
                      n (.count claccessor buf) (class buf))))))
  (create-matrix [this m n buf order]
    (if (and (<= 0 (* (long m) (long n)) (.count claccessor buf))
             (instance? CLBuffer buf))
      (let [order (or order DEFAULT_ORDER)
            ld (max (long (if (= COLUMN_MAJOR order) m n)) 1)]
        (->CLGeneralMatrix this claccessor
                           (matrix-engine this [buf m n 0 ld order])
                           (.entryType claccessor) true buf m n 0 ld order))
      (throw (IllegalArgumentException.
              (format "I do not know how to create a %dx%d matrix from %s."
                      m n (type buf))))))
  (data-accessor [_]
    claccessor)
  (vector-engine [_ [cl-buf n ofst strd]]
    (let [iacc-size (* Integer/BYTES (count-work-groups WGS n))
          acc-size (* Double/BYTES (count-work-groups WGS n))
          cl-acc (cl-buffer ctx acc-size :read-write)
          cl-iacc (cl-buffer ctx iacc-size :read-write)
          cl-eq-flag (cl-buffer ctx Integer/BYTES :read-write)
          cl-ofst (wrap-int ofst)
          cl-strd (wrap-int strd)]
      (->GCNVectorEngine WGS
                         claccessor
                         queue
                         cl-buf
                         n ofst strd
                         cl-eq-flag
                         cl-acc
                         cl-iacc
                         (work-size-1d n)
                         (doto (kernel prog "equals_vector")
                           (set-args! 1 cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "swap")
                           (set-args! 0 cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "copy")
                           (set-args! 0 cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "scal")
                           (set-args! 1 cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "axpy")
                           (set-args! 1 cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "sum_reduction")
                           (set-args! cl-acc))
                         (doto (kernel prog "imax_reduction")
                           (set-args! cl-iacc cl-acc))
                         (doto (kernel prog "dot_reduce")
                           (set-args! 0 cl-acc cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "nrm2_reduce")
                           (set-args! 0 cl-acc cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "asum_reduce")
                           (set-args! 0 cl-acc cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "iamax_reduce")
                           (set-args! 0 cl-iacc cl-acc cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "sum_reduce")
                           (set-args! 0 cl-acc cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "imax_reduce")
                           (set-args! 0 cl-iacc cl-acc cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "imin_reduce")
                           (set-args! 0 cl-iacc cl-acc cl-buf cl-ofst cl-strd)))))
  (matrix-engine [_ [cl-buf m n ofst ld ord]]
    (let [acc-size (* (.entryWidth ^DataAccessor claccessor)
                      (max 1 (* (long m) (count-work-groups WGSn n))))
          cl-acc (cl-buffer ctx acc-size :read-write)
          cl-eq-flag (cl-buffer ctx Integer/BYTES :read-write)
          cl-ofst (wrap-int ofst)
          cl-ld (wrap-int ld)
          cl-ord (wrap-int ord)]
      (->GCNMatrixEngine WGSn TS WPT
                         claccessor queue
                         m n ofst ld ord
                         cl-eq-flag
                         cl-acc
                         (work-size-1d m)
                         (doto (kernel prog "equals_matrix")
                           (set-args! 1 cl-buf cl-ofst cl-ld))
                         (doto (kernel prog "axpby")
                           (set-args! 0 (wrap-prim claccessor 1) cl-acc
                                      (wrap-int 0) (wrap-int 1)))
                         (doto (kernel prog "sum_reduction_horizontal")
                           (set-arg! 0 cl-acc))
                         (doto (kernel prog "copy_matrix")
                           (set-args! 0 cl-buf cl-ofst cl-ld))
                         (doto (kernel prog "swap_matrix")
                           (set-args! 0 cl-buf cl-ofst cl-ld))
                         (doto (kernel prog "gerk")
                           (set-args! 7 cl-buf cl-ofst cl-ld))
                         (doto (kernel prog "gemv_reduce")
                           (set-arg! 0 cl-acc)
                           (set-args! 2 cl-buf cl-ofst cl-ld))
                         (doto (kernel prog "gemm_tiled")
                           (set-args! 1 cl-buf cl-ofst cl-ld))
                         (doto (kernel prog "gemm_tiled_fit")
                           (set-args! 1 cl-buf cl-ofst cl-ld))))))

(let [src [(slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
           (slurp (io/resource "uncomplicate/neanderthal/opencl/kernels/amd_gcn/blas.cl"))]]
  (defn gcn-factory
    ([create-accessor ctx queue wgs wgsn ts wpt]
     (let [accessor (create-accessor ctx queue)]
       (->GCNFactory
        accessor ctx queue
        (build-program!
         (program-with-source ctx src)
         (format "-cl-std=CL2.0 -DNUMBER=%s -DACCUMULATOR=%s -DREAL=%s -DWGS=%d -DWGSn=%d -DTS=%d -DWPT=%d"
                 (.entryType ^DataAccessor accessor) Double/TYPE
                 (.entryType ^DataAccessor accessor) wgs wgsn ts wpt)
         nil)
        wgs wgsn ts wpt)))
    ([create-accessor ctx queue]
     (gcn-factory create-accessor ctx queue 256 16 32 4))))
