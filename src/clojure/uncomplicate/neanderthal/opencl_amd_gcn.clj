(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl-amd-gcn
  (:require [uncomplicate.neanderthal
             [opencl :refer :all]
             [math :refer [sqrt]]]
            [uncomplicate.clojurecl.core :refer :all])
  (:import [uncomplicate.clojurecl.core WorkSize]
           [uncomplicate.neanderthal.protocols BLAS BLASPlus Block Matrix]))

(defn ^:private count-work-groups ^long [^long max-local-size ^long n]
  (if (< max-local-size n)
    (quot (+ n (dec max-local-size)) max-local-size)
    1))

(defn ^:private enq-reduce
  [queue main-kernel reduce-kernel max-local-size n]
  (loop [queue (enq-nd! queue main-kernel (work-size [n]))
         global-size (count-work-groups max-local-size n)]
    (if (= 1 global-size)
      queue
      (recur
       (enq-nd! queue reduce-kernel (work-size [global-size]))
       (count-work-groups max-local-size global-size)))))

(defn ^:private enq-reduce-horizontal
  [queue main-kernel reduce-kernel max-local-size m n]
  (loop [queue (enq-nd! queue main-kernel (work-size [m n]))
         folded-n (count-work-groups max-local-size n)]
    (if (= 1 folded-n)
      queue
      (recur
       (enq-nd! queue reduce-kernel (work-size [m folded-n]))
       (count-work-groups max-local-size folded-n)))))

(defn ^:private enq-read-int ^long [queue cl-buf]
  (let [res (int-array 1)]
    (enq-read! queue cl-buf res)
    (aget res 0)))

(defn ^:private enq-read-double ^double [queue cl-buf]
  (let [res (double-array 1)]
    (enq-read! queue cl-buf res)
    (aget res 0)))

(deftype GCNVectorEngine [^long WGS
                          claccessor
                          queue
                          cl-buf
                          n
                          reduce-acc
                          reduce-iacc
                          linear-work-size
                          swp-kernel
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
     (release reduce-acc)
     (release reduce-iacc)
     (release swp-kernel)
     (release scal-kernel)
     (release axpy-kernel)
     (release sum-reduction-kernel)
     (release imax-reduction-kernel)
     (release dot-reduce-kernel)
     (release nrm2-reduce-kernel)
     (release asum-reduce-kernel)
     (release iamax-reduce-kernel)
     (release sum-reduce-kernel)))
  BLAS
  (swap [_ _ y]
    (do
      (set-arg! swp-kernel 1 (.buffer y))
      (enq-nd! queue swp-kernel linear-work-size)))
  (copy [_ _ y]
    (enq-copy! cl-buf (.buffer ^Block y)))
  (dot [_ _ y]
    (do
      (set-arg! dot-reduce-kernel 2 (.buffer y))
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
      (set-arg! scal-kernel 0 (array claccessor [alpha]))
      (enq-nd! queue scal-kernel linear-work-size)))
  (axpy [_ alpha x y]
    (do
      (set-arg! axpy-kernel 0 (array claccessor [alpha]))
      (set-arg! axpy-kernel 2 (.buffer y))
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
                          m n
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
      (set-args! axpby-kernel 2 (array claccessor [beta]) (.buffer y))
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
  (cl-accessor [_]
    claccessor)
  (vector-engine [_ cl-buf n]
    (let [iacc-size (* Integer/BYTES (count-work-groups WGS n))
          acc-size (* Double/BYTES (count-work-groups WGS n))
          cl-acc (cl-buffer ctx acc-size :read-write)
          cl-iacc (cl-buffer ctx iacc-size :read-write)]
      (->GCNVectorEngine WGS
                         claccessor
                         queue
                         cl-buf
                         n
                         cl-acc
                         cl-iacc
                         (work-size [n])
                         (doto (kernel prog "swp") (set-arg! 0 cl-buf))
                         (doto (kernel prog "scal") (set-arg! 1 cl-buf))
                         (doto (kernel prog "axpy") (set-arg! 1 cl-buf))
                         (doto (kernel prog "sum_reduction")
                           (set-args! cl-acc))
                         (doto (kernel prog "imax_reduction")
                           (set-args! cl-iacc cl-acc))
                         (doto (kernel prog "dot_reduce")
                           (set-args! cl-acc cl-buf))
                         (doto (kernel prog "nrm2_reduce")
                           (set-args! cl-acc cl-buf))
                         (doto (kernel prog "asum_reduce")
                           (set-args! cl-acc cl-buf))
                         (doto (kernel prog "iamax_reduce")
                           (set-args! cl-iacc cl-acc cl-buf))
                         (doto (kernel prog "sum_reduce")
                           (set-args! cl-acc cl-buf)))))
  (matrix-engine [_ cl-buf m n]
    (let [acc-size (* (long (width claccessor)) (long m) (count-work-groups WGSn n))
          cl-acc (cl-buffer ctx acc-size :read-write)]
      (->GCNMatrixEngine WGSn TS WPT
                         claccessor
                         queue
                         m n
                         cl-acc
                         (work-size [m])
                         (doto (kernel prog "axpby")
                           (set-args! 0 (array claccessor [1]) cl-acc))
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
        ctx [(slurp "resources/opencl/blas.cl")])
       (format "-cl-std=CL2.0 -DREAL=%s -DWGS=%d -DWGSn=%d -DTS=%d -DWPT=%d"
               (entryType accessor) wgs wgsn ts wpt)
       nil)
      wgs wgsn ts wpt)))
  ([create-accessor ctx queue]
   (gcn-engine-factory create-accessor ctx queue 256 16 32 4)))

(defn gcn-single [ctx queue]
  (gcn-engine-factory float-accessor ctx queue))

(defn gcn-double [ctx queue]
  (gcn-engine-factory double-accessor ctx queue))
