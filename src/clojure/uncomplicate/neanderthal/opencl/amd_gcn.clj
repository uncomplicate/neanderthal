(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl.amd-gcn
  (:refer-clojure :exclude [accessor])
  (:require [clojure.java.io :as io]
            [uncomplicate.clojurecl.core :refer :all]
            [uncomplicate.clojurecl.toolbox
             :refer [count-work-groups enq-reduce enq-reduce-horizontal
                     enq-read-int enq-read-double wrap-int]]
            [uncomplicate.neanderthal.protocols :refer :all]
            [uncomplicate.neanderthal.math :refer [sqrt]]
            [uncomplicate.neanderthal.opencl.clblock :refer :all])
  (:import [uncomplicate.clojurecl.core WorkSize]
           [uncomplicate.neanderthal.protocols
            BLAS BLASPlus Block Matrix DataAccessor]))

(deftype GCNVectorEngine [^long WGS
                          claccessor
                          queue
                          cl-buf
                          ^long n
                          ^long ofst
                          ^long strd
                          eq-flag
                          reduce-acc
                          reduce-iacc
                          linear-work-size
                          equals-vector-kernel
                          swp-kernel
                          copy-kernel
                          scal-kernel
                          axpy-kernel
                          sum-reduction-kernel
                          imax-reduction-kernel
                          dot-reduce-kernel
                          nrm2-reduce-kernel
                          asum-reduce-kernel
                          iamax-reduce-kernel
                          sum-reduce-kernel]

  Releaseable
  (release [_]
    (and
     (release eq-flag)
     (release reduce-acc)
     (release reduce-iacc)
     (release equals-vector-kernel)
     (release swp-kernel)
     (release copy-kernel)
     (release scal-kernel)
     (release axpy-kernel)
     (release sum-reduction-kernel)
     (release imax-reduction-kernel)
     (release dot-reduce-kernel)
     (release nrm2-reduce-kernel)
     (release asum-reduce-kernel)
     (release iamax-reduce-kernel)
     (release sum-reduce-kernel)))
  BlockEngine
  (equals-vector [_ _ y]
    (let [res (wrap-int 0)]
      (set-args! equals-vector-kernel 4 (.buffer ^Block y)
                 (wrap-int (.offset ^Block y)) (wrap-int (.stride ^Block y)))
      (enq-fill! queue eq-flag res)
      (set-arg! equals-vector-kernel 0 eq-flag)
      (enq-nd! queue equals-vector-kernel linear-work-size)
      (enq-read! queue eq-flag res)
      (= 0 (aget res 0))))
  BLAS
  (swap [_ _ y]
    (do
      (set-args! swp-kernel 3 (.buffer y) (wrap-int (.offset y))
                 (wrap-int (.stride y)))
      (enq-nd! queue swp-kernel linear-work-size)))
  (copy [_ x y]
    (if (and (= 0 strd) (= 1 strd (.stride y)))
      (enq-copy! cl-buf (.buffer y))
      (do
        (set-args! copy-kernel 3 (.buffer y) (wrap-int (.offset y))
                   (wrap-int (.stride y)))
        (enq-nd! queue copy-kernel linear-work-size))))
  (dot [_ _ y]
    (do
      (set-args! dot-reduce-kernel 4 (.buffer y) (wrap-int (.offset y))
                (wrap-int (.stride y)))
      (enq-reduce queue dot-reduce-kernel sum-reduction-kernel WGS n)
      (enq-read-double queue reduce-acc)))
  (nrm2 [_ _]
    (do
      (enq-reduce queue nrm2-reduce-kernel sum-reduction-kernel WGS n)
      (sqrt (enq-read-double queue reduce-acc))))
  (asum [_ _]
    (do
      (enq-reduce queue asum-reduce-kernel sum-reduction-kernel WGS n)
      (enq-read-double queue reduce-acc)))
  (iamax [_ _]
    (do
      (enq-reduce queue iamax-reduce-kernel imax-reduction-kernel WGS n)
      (enq-read-int queue reduce-iacc)))
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
  (scal [_ alpha _]
    (do
      (set-args! scal-kernel 0 (array claccessor [alpha]))
      (enq-nd! queue scal-kernel linear-work-size)))
  (axpy [_ alpha x y]
    (do
      (set-args! axpy-kernel 0 (array claccessor [alpha]))
      (set-args! axpy-kernel 4 (.buffer y) (wrap-int (.offset y))
                (wrap-int (.stride y)))
      (enq-nd! queue axpy-kernel linear-work-size)))
  BLASPlus
  (sum [_ x]
    (do
      (enq-reduce queue sum-reduce-kernel sum-reduction-kernel WGS n)
      (enq-read-double queue reduce-acc))))

;; ======================= Dense Matrix ========================================

(deftype GCNMatrixEngine [^long WGSn
                          ^long TS
                          ^long WPT
                          claccessor
                          queue
                          ^long m ^long n ^long ld
                          reduce-acc
                          linear-work-size
                          axpby-kernel
                          sum-reduction-horizontal-kernel
                          gemv-reduce-kernel
                          gemm-tiled-kernel
                          gemm-tiled-fit-kernel]
  Releaseable
  (release [_]
    (and
     (release reduce-acc)
     (release axpby-kernel)
     (release sum-reduction-horizontal-kernel)
     (release gemv-reduce-kernel)
     (release gemm-tiled-kernel)
     (release gemm-tiled-fit-kernel)))
  BLAS
  (mv [_ alpha _ x beta y]
    (do
      (set-arg! gemv-reduce-kernel 1 (array claccessor [alpha]))
      (set-arg! gemv-reduce-kernel 3 (.buffer x))
      (enq-reduce-horizontal queue gemv-reduce-kernel
                             sum-reduction-horizontal-kernel
                             WGSn m n)
      (set-args! axpby-kernel 4 (array claccessor [beta]) (.buffer y)
                 (wrap-int (.offset y)) (wrap-int (.stride y)))
      (enq-nd! queue axpby-kernel linear-work-size)))
  (mm [_ alpha a b beta c]
    (let [cn (/ (.ncols ^Matrix b) WPT)
          gemm-kernel (if (= 0 (mod m TS) (mod cn TS))
                        gemm-tiled-fit-kernel
                        gemm-tiled-kernel)]
      (set-arg! gemm-kernel 0 (array claccessor [alpha]))
      (set-args! gemm-kernel 2
                 (.buffer b) (array claccessor [beta]) (.buffer c)
                 (int-array [m]) (int-array [n]) (int-array [(.ncols ^Matrix b)]))
      (enq-nd! queue gemm-kernel
               (work-size [(* TS (count-work-groups TS m))
                           (* TS (count-work-groups TS cn))])))))

(deftype GCNEngineFactory [claccessor ctx queue prog ^long WGS WGSn TS WPT]
  Releaseable
  (release [_]
    (release prog))
  EngineFactory
  (data-accessor [_]
    claccessor)
  (vector-engine [_ cl-buf n ofst strd]
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
                         (work-size [n])
                         (doto (kernel prog "equals_vector")
                           (set-args! 1 cl-buf cl-ofst cl-strd))
                         (doto (kernel prog "swp")
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
                           (set-args! 0 cl-acc cl-buf cl-ofst cl-strd)))))
  (matrix-engine [_ cl-buf m n ld]
    (let [acc-size (* (.entryWidth ^DataAccessor claccessor)
                      (long m) (count-work-groups WGSn n))
          cl-acc (cl-buffer ctx acc-size :read-write)]
      (->GCNMatrixEngine WGSn TS WPT
                         claccessor
                         queue
                         m n ld
                         cl-acc
                         (work-size [m])
                         (doto (kernel prog "axpby")
                           (set-args! 0 (array claccessor [1]) cl-acc
                                      (wrap-int 0) (wrap-int 1)))
                         (doto (kernel prog "sum_reduction_horizontal")
                           (set-arg! 0 cl-acc))
                         (doto (kernel prog "gemv_reduce")
                           (set-arg! 0 cl-acc)
                           (set-arg! 2 cl-buf))
                         (doto (kernel prog "gemm_tiled")
                           (set-arg! 1 cl-buf))
                         (doto (kernel prog "gemm_tiled_fit")
                           (set-arg! 1 cl-buf))))))

(defn gcn-engine-factory
  ([create-accessor ctx queue wgs wgsn ts wpt]
   (let [accessor (create-accessor ctx queue)]
     (->GCNEngineFactory
      accessor ctx queue
      (build-program!
       (program-with-source
        ctx [(slurp (io/resource "uncomplicate/clojurecl/kernels/reduction.cl"))
             (slurp (io/resource "uncomplicate/neanderthal/opencl/kernels/amd_gcn/blas.cl"))])
       (format "-cl-std=CL2.0 -DNUMBER=%s -DACCUMULATOR=%s -DREAL=%s -DWGS=%d -DWGSn=%d -DTS=%d -DWPT=%d"
               (.entryType ^DataAccessor accessor) Double/TYPE
               (.entryType ^DataAccessor accessor) wgs wgsn ts wpt)
       nil)
      wgs wgsn ts wpt)))
  ([create-accessor ctx queue]
   (gcn-engine-factory create-accessor ctx queue 256 16 32 4)))
