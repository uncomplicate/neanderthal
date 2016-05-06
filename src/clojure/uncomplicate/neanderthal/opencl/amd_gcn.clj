(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.amd-gcn
  (:refer-clojure :exclude [accessor])
  (:require [clojure.java.io :as io]
            [uncomplicate.commons.core
             :refer [Releaseable release with-release wrap-int]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [toolbox :refer [count-work-groups enq-reduce
                              enq-read-int enq-read-double]]]
            [uncomplicate.neanderthal
             [protocols :refer :all]
             [block :refer :all]
             [math :refer [sqrt]]
             [core :refer [dim mrows ncols]]]
            [uncomplicate.neanderthal.opencl.clblock :refer :all])
  (:import [uncomplicate.clojurecl.core CLBuffer]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Block Matrix DataAccessor]))

(deftype GCNVectorEngine [ctx queue prog claccessor ^long WGS ]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ x y]
    (if (< 0 (dim x))
      (with-release [equals-vector-kernel (kernel prog "equals_vector")
                     eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
        (let [res (wrap-int 0)]
          (enq-fill! queue eq-flag-buf res)
          (set-args! equals-vector-kernel eq-flag-buf
                     (.buffer ^Block x)
                     (wrap-int (.offset ^Block x)) (wrap-int (.stride ^Block x))
                     (.buffer ^Block y)
                     (wrap-int (.offset ^Block y)) (wrap-int (.stride ^Block y)))
          (enq-nd! queue equals-vector-kernel (work-size-1d (dim x)))
          (enq-read! queue eq-flag-buf res)
          (= 0 (aget res 0))))
      true))
  BLAS
  (swap [_ x y]
    (if (< 0 (dim x))
      (with-release [swap-kernel (kernel prog "swap")]
        (set-args! swap-kernel
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-nd! queue swap-kernel (work-size-1d (dim x))))
      queue))
  (copy [_ x y]
    (if (< 0 (dim x))
      (with-release [copy-kernel (kernel prog "copy")]
        (set-args! copy-kernel
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-nd! queue copy-kernel (work-size-1d (dim x))))
      queue))
  (dot [_ x y]
    (if (< 0 (dim x))
      (with-release [dot-reduce-kernel (kernel prog "dot_reduce")
                     sum-reduction-kernel (kernel prog "sum_reduction")
                     reduce-acc
                     (cl-buffer ctx
                                (* Double/BYTES (count-work-groups WGS (dim x)))
                                :read-write)]
        (set-args! sum-reduction-kernel reduce-acc)
        (set-args! dot-reduce-kernel reduce-acc
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-reduce queue dot-reduce-kernel sum-reduction-kernel (dim x) WGS)
        (enq-read-double queue reduce-acc))
      0.0))
  (nrm2 [_ x]
    (if (< 0 (dim x))
      (with-release [nrm2-reduce-kernel (kernel prog "nrm2_reduce")
                     sum-reduction-kernel (kernel prog "sum_reduction")
                     reduce-acc
                     (cl-buffer ctx
                                (* Double/BYTES (count-work-groups WGS (dim x)))
                                :read-write)]
        (set-args! sum-reduction-kernel reduce-acc)
        (set-args! nrm2-reduce-kernel
                   reduce-acc (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
        (enq-reduce queue nrm2-reduce-kernel sum-reduction-kernel (dim x) WGS)
        (sqrt (enq-read-double queue reduce-acc)))
      0.0))
  (asum [_ x]
    (if (< 0 (dim x))
      (with-release [asum-reduce-kernel (kernel prog "asum_reduce")
                     sum-reduction-kernel (kernel prog "sum_reduction")
                     reduce-acc
                     (cl-buffer ctx
                                (* Double/BYTES (count-work-groups WGS (dim x)))
                                :read-write)]
        (set-args! sum-reduction-kernel reduce-acc)
        (set-args! asum-reduce-kernel
                   reduce-acc (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
        (enq-reduce queue asum-reduce-kernel sum-reduction-kernel (dim x) WGS)
        (enq-read-double queue reduce-acc))
      0.0))
  (iamax [_ x]
    (if (< 0 (dim x))
      (let [wgcount (count-work-groups WGS (dim x))]
        (with-release [iamax-reduce-kernel (kernel prog "iamax_reduce")
                       imax-reduction-kernel (kernel prog "imax_reduction")
                       reduce-acc (cl-buffer ctx (* Double/BYTES wgcount) :read-write)
                       reduce-iacc (cl-buffer ctx (* Integer/BYTES wgcount) :read-write)]
          (set-args! imax-reduction-kernel reduce-iacc reduce-acc)
          (set-args! iamax-reduce-kernel reduce-iacc reduce-acc
                     (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
          (enq-reduce queue iamax-reduce-kernel imax-reduction-kernel (dim x) WGS)
          (enq-read-int queue reduce-iacc)))
      0))
  (rot [_ _ y c s]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotg [_ _]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotm [_ _ y p]
    (throw (UnsupportedOperationException. "TODO.")))
  (rotmg [_ _ args]
    (throw (UnsupportedOperationException. "TODO.")))
  (scal [_ alpha x]
    (if (< 0 (dim x))
      (with-release [scal-kernel (kernel prog "scal")]
        (set-args! scal-kernel
                   (wrap-prim claccessor alpha)
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
        (enq-nd! queue scal-kernel (work-size-1d (dim x))))
      queue))
  (axpy [_ alpha x y]
    (if (< 0 (dim x))
      (with-release [axpy-kernel (kernel prog "axpy")]
        (set-args! axpy-kernel
                   (wrap-prim claccessor alpha)
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-nd! queue axpy-kernel (work-size-1d (dim x))))
      queue))
  BLASPlus
  (subcopy [_ x y kx lx ky]
    (if (< 0 lx)
      (with-release [copy-kernel (kernel prog "copy")]
        (set-args! copy-kernel
                   (.buffer x) (wrap-int (+ (.offset x) kx)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (+ (.offset y) ky)) (wrap-int (.stride y)))
        (enq-nd! queue copy-kernel (work-size-1d lx)))
      queue))
  (sum [_ x]
    (if (< 0 (dim x))
      (with-release [sum-reduce-kernel (kernel prog "sum_reduce")
                     sum-reduction-kernel (kernel prog "sum_reduction")
                     reduce-acc
                     (cl-buffer ctx
                                (* Double/BYTES (count-work-groups WGS (dim x)))
                                :read-write)]
        (set-args! sum-reduction-kernel reduce-acc)
        (set-args! sum-reduce-kernel
                   reduce-acc (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
        (enq-reduce queue sum-reduce-kernel sum-reduction-kernel (dim x) WGS)
        (enq-read-double queue reduce-acc))
      0.0))
  (imax [_ x]
    (if (< 0 (dim x))
      (let [wgcount (count-work-groups WGS (dim x))]
        (with-release [imax-reduce-kernel (kernel prog "imax_reduce")
                       imax-reduction-kernel (kernel prog "imax_reduction")
                       reduce-acc (cl-buffer ctx (* Double/BYTES wgcount) :read-write)
                       reduce-iacc (cl-buffer ctx (* Integer/BYTES wgcount) :read-write)]
          (set-args! imax-reduction-kernel reduce-iacc reduce-acc)
          (set-args! imax-reduce-kernel reduce-iacc reduce-acc
                     (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
          (enq-reduce queue imax-reduce-kernel imax-reduction-kernel (dim x) WGS)
          (enq-read-int queue reduce-iacc)))
      0))
  (imin [_ x]
    (if (< 0 (dim x))
      (let [wgcount (count-work-groups WGS (dim x))]
        (with-release [imin-reduce-kernel (kernel prog "imin_reduce")
                       imax-reduction-kernel (kernel prog "imax_reduction")
                       reduce-acc (cl-buffer ctx (* Double/BYTES wgcount) :read-write)
                       reduce-iacc (cl-buffer ctx (* Integer/BYTES wgcount) :read-write)]
          (set-args! imax-reduction-kernel reduce-iacc reduce-acc)
          (set-args! imin-reduce-kernel reduce-iacc reduce-acc
                     (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
          (enq-reduce queue imin-reduce-kernel imax-reduction-kernel (dim x) WGS)
          (enq-read-int queue reduce-iacc)))
      0)))

;; ======================= Dense Matrix ========================================

(deftype GCNMatrixEngine [ctx queue prog claccessor
                          ^long WGSm ^long WGSn ^long TS ^long WPT]
  Releaseable
  (release [_]
    true)
  BlockEngine
  (equals-block [_ a b]
    (with-release [equals-matrix-kernel (kernel prog "equals_matrix")
                   eq-flag-buf (cl-buffer ctx Integer/BYTES :read-write)]
      (let [res (wrap-int 0)]
        (enq-fill! queue eq-flag-buf res)
        (set-args! equals-matrix-kernel eq-flag-buf
                   (.buffer ^Block a)
                   (wrap-int (.offset ^Block a)) (wrap-int (.stride ^Block a))
                   (.buffer ^Block b)
                   (wrap-int (.offset ^Block b)) (wrap-int (.stride ^Block b)))
        (enq-nd! queue equals-matrix-kernel (work-size-2d (mrows a) (ncols a)))
        (enq-read! queue eq-flag-buf res)
        (= 0 (aget res 0)))))
  BLAS
  (swap [_ a b]
    (if (< 0 (* (mrows a) (ncols a)))
      (with-release [swap-matrix-kernel (kernel prog "swap_matrix")]
        (set-args! swap-matrix-kernel
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
        (enq-nd! queue swap-matrix-kernel (work-size-2d (mrows a) (ncols a)))))
    queue)
  (copy [_ a b]
    (if (< 0 (* (mrows a) (ncols a)))
      (with-release [copy-matrix-kernel (kernel prog "copy_matrix")]
        (set-args! copy-matrix-kernel
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b)))
        (enq-nd! queue copy-matrix-kernel (work-size-2d (mrows a) (ncols b))))
      queue))
  (rank [_ alpha x y a]
    (if (< 0 (* (mrows a) (ncols a)))
      (with-release [gerk-kernel (kernel prog "gerk")]
        (set-args! gerk-kernel 0 (wrap-prim claccessor alpha)
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x))
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y))
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a)))
        (enq-nd! queue gerk-kernel (work-size-2d (mrows a) (ncols a))))
      queue))
  (mv [_ alpha a x beta y]
    (let [acc-size (* (.entryWidth ^DataAccessor claccessor)
                      (max 1 (* (mrows a) (count-work-groups WGSn (ncols a)))))]
      (with-release [gemv-reduce-kernel (kernel prog "gemv_reduce")
                     sum-reduction-horizontal-kernel
                     (kernel prog "sum_reduction_horizontal")
                     axpby-kernel (kernel prog "axpby")
                     reduction-acc (cl-buffer ctx acc-size :read-write)]
        (set-arg! sum-reduction-horizontal-kernel 0 reduction-acc)
        (set-args! gemv-reduce-kernel reduction-acc (wrap-prim claccessor alpha)
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.buffer x) (wrap-int (.offset x)) (wrap-int (.stride x)))
        (enq-reduce queue gemv-reduce-kernel sum-reduction-horizontal-kernel
                    (mrows a) (ncols a) WGSm WGSn)
        (set-args! axpby-kernel
                   (wrap-prim claccessor 1)
                   reduction-acc (wrap-int 0) (wrap-int 1)
                   (wrap-prim claccessor beta)
                   (.buffer y) (wrap-int (.offset y)) (wrap-int (.stride y)))
        (enq-nd! queue axpby-kernel (work-size-1d (mrows a))))))
  (mm [_ alpha a b beta c]
    (let [m (mrows a)
          k (ncols a)
          n (ncols b)
          RTS (long (/ TS WPT))]
      (with-release [gemm-kernel (if (= 0 (mod m TS) (mod n TS))
                                   (kernel prog "gemm_tiled_fit")
                                   (kernel prog "gemm_tiled"))]
        (set-args! gemm-kernel
                   (wrap-prim claccessor alpha)
                   (.buffer a) (wrap-int (.offset a)) (wrap-int (.stride a))
                   (.buffer b) (wrap-int (.offset b)) (wrap-int (.stride b))
                   (wrap-prim claccessor beta)
                   (.buffer c) (wrap-int (.offset c)) (wrap-int (.stride c))
                   (wrap-int m) (wrap-int k) (wrap-int n))
        (enq-nd! queue gemm-kernel
                 (work-size-2d (* TS (count-work-groups TS m))
                               (* RTS (count-work-groups TS n))))))))

(deftype GCNFactory [ctx queue prog ^DataAccessor claccessor vector-eng matrix-eng]
  Releaseable
  (release [_]
    (and
     (release prog)
     (release vector-eng)
     (release matrix-eng)))
  FactoryProvider
  (factory [_]
    (factory claccessor))
  Factory
  (create-vector [this n buf _]
    (create-cl-vector this vector-eng n buf))
  (create-matrix [this m n buf ord]
    (create-cl-ge-matrix this matrix-eng m n buf ord))
  (data-accessor [_]
    claccessor)
  (vector-engine [_]
    vector-eng)
  (matrix-engine [_]
    matrix-eng))

(let [src [(slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
           (slurp (io/resource "uncomplicate/neanderthal/opencl/kernels/amd_gcn/blas.cl"))]]

  (defn gcn-factory
    ([create-accessor ctx queue wgs wgsm wgsn ts wpt]
     (let [accessor (create-accessor ctx queue)
           prog (build-program!
                 (program-with-source ctx src)
                 (format "-cl-std=CL2.0 -DNUMBER=%s -DACCUMULATOR=%s -DREAL=%s -DWGS=%d -DWGSm=%d -DWGSn=%d -DTS=%d -DWPT=%d"
                         (.entryType ^DataAccessor accessor) Double/TYPE
                         (.entryType ^DataAccessor accessor) wgs wgsm wgsn ts wpt)
                 nil)]
       (->GCNFactory
        ctx queue prog accessor
        (->GCNVectorEngine ctx queue prog accessor wgs)
        (->GCNMatrixEngine ctx queue prog accessor wgsm wgsn ts wpt ))))
    ([create-accessor ctx queue]
     (gcn-factory create-accessor ctx queue 256 16 16 32 4)))

  (defn gcn-single [ctx queue]
    (gcn-factory cl-float-accessor ctx queue))

  (defn gcn-double [ctx queue]
    (gcn-factory cl-double-accessor ctx queue)))
