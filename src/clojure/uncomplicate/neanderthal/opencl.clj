(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.neanderthal.math :refer [sqrt]]
   [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [max-work-group-size queue-device queue-context]]])
  (:import [uncomplicate.neanderthal.protocols
            Vector RealVector Matrix RealMatrix Carrier RealChangeable]
           [uncomplicate.clojurecl.core WorkSize]
           [uncomplicate.neanderthal.cblas FloatBlockVector]
           [java.nio ByteBuffer]))

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

(defn ^:private enq-read-int ^long [queue cl-buf]
  (let [res (int-array 1)]
    (enq-read! queue cl-buf res)
    (aget res 0)))

(defn ^:private enq-read-double ^double [queue cl-buf]
  (let [res (double-array 1)]
    (enq-read! queue cl-buf res)
    (aget res 0)))

(defprotocol Mappable
  (read! [this v])
  (write! [this v])
  (map-host [this])
  (unmap [this]))

;;TODO Switch to these (as interfaces) for ATLAS, too, and rename methods then.
(defprotocol BLAS1
    (s-iamax [_ cl-x])
    (s-rotg [_ cl-x])
    (s-rotm [_ cl-x cl-y p])
    (s-rotmg [_ p args])
    (s-dot [_ cl-x cl-y])
    (s-nrm2 [_ cl-x])
    (s-asum [_ cl-x])
    (s-rot [_ cl-x cl-y c s])
    (s-scal [_ alpha cl-x])
    (s-axpy [_ alpha cl-x cl-y])
    (s-swp [_ cl-x cl-y]))

(deftype GCNVectorEngine [^long entry-width
                          ^long max-local-size
                          queue
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
                          iamax-reduce-kernel]

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
     (release iamax-reduce-kernel)))

  BLAS1
  (s-iamax [_ _]
    (do
      (enq-reduce queue iamax-reduce-kernel imax-reduction-kernel
                  max-local-size n)
      (enq-read-int queue reduce-iacc)))
  (s-rotg [_ _]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (s-rotm [_ _ cl-y p]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (s-rotmg [_ p args]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (s-dot [_ _ cl-y]
    (do
      (set-arg! dot-reduce-kernel 2 cl-y)
      (enq-reduce queue dot-reduce-kernel sum-reduction-kernel max-local-size n)
      (enq-read-double queue reduce-acc)))
  (s-nrm2 [_ _]
    (do
      (enq-reduce queue nrm2-reduce-kernel sum-reduction-kernel max-local-size n)
      (sqrt (enq-read-double queue reduce-acc))))
  (s-asum [_ _]
    (do
      (enq-reduce queue asum-reduce-kernel sum-reduction-kernel max-local-size n)
        (enq-read-double queue reduce-acc)))
  (s-rot [_ _ cl-y c s]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (s-scal [_ alpha _]
    (do
      (set-arg! scal-kernel 0 (float-array [alpha]))
      (enq-nd! queue scal-kernel linear-work-size)))
  (s-axpy [_ alpha _ cl-y]
    (do
      (set-arg! axpy-kernel 0 (float-array [alpha]))
      (set-arg! axpy-kernel 2 cl-y)
      (enq-nd! queue axpy-kernel linear-work-size)))
  (s-swp [_ _ cl-y]
    (do
      (set-arg! swp-kernel 1 cl-y)
      (enq-nd! queue swp-kernel linear-work-size))))

(defn gcn-vector-engine
  [context queue prog max-local-size cl-buf entry-width n]
  (let [iacc-size (* Integer/BYTES (count-work-groups max-local-size n))
        acc-size (* Double/BYTES (count-work-groups max-local-size n))
        cl-acc (cl-buffer context acc-size :read-write)
        cl-iacc (cl-buffer context iacc-size :read-write)]
    (->GCNVectorEngine entry-width
                       max-local-size
                       queue
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
                         (set-args! cl-iacc cl-acc cl-buf)))))

(defrecord CLSettings [^long max-local-size queue context prog]
  Releaseable
  (release [_]
    (release prog)))

(defn cl-settings [queue]
  (let [dev (queue-device queue)
        ctx (queue-context queue)
        prog (build-program! (program-with-source
                              ctx [(slurp "resources/opencl/blas.cl")])
                             "-cl-std=CL2.0" nil)]
    (->CLSettings (max-work-group-size dev) queue ctx prog)))

(def zero-array (float-array [0]))

(declare cl-real-vector)

(deftype RealVectorCL [^CLSettings settings engine cl-buf ^long width-bytes ^long n]

  Releaseable
  (release [_]
    (and
     (release cl-buf)
     (release engine)))

  Vector
  (dim [_]
    n)
  (subvector [_ k l]
    (cl-real-vector settings
                    (cl-sub-buffer cl-buf (* width-bytes k) (* width-bytes l))
                    width-bytes
                    l))
  (iamax [_]
    (s-iamax engine cl-buf))
  (rotg [_]
    (s-rotg engine cl-buf))
  (rotm [_ y p]
    (s-rotm engine cl-buf (.cl-buf ^RealVectorCL y) p))
  (rotmg [p args]
    (s-rotmg engine p args))
  RealVector
  (entry [_ i]
    (throw (UnsupportedOperationException.
            "To access values from the CL device, read! them first.")))
  (dot [_ y]
    (s-dot engine cl-buf (.cl-buf ^RealVectorCL y)))
  (nrm2 [_]
    (s-nrm2 engine cl-buf))
  (asum [_]
    (s-asum engine cl-buf))
  (rot [_ y c s]
    (s-rot engine cl-buf (.cl-buf ^RealVectorCL y) c s))
  (scal [x alpha]
    (do
      (s-scal engine alpha cl-buf)
      x))
  (axpy [_ alpha y]
    (do
      (s-axpy engine alpha cl-buf (.cl-buf ^RealVectorCL y))
      y))
  RealChangeable
  (set [x val]
    (do
      (enq-fill! (.queue settings) cl-buf (float-array [val]))
      x))
  Mappable
  (read! [_ host]
    (do
      (enq-read! (.queue settings) cl-buf (.buf ^FloatBlockVector host))
      host))
  (write! [this host]
    (do
      (enq-write! (.queue settings) cl-buf (.buf ^FloatBlockVector host))
      this))
  Carrier
  (zero [_]
    (let [zero-sv ^RealVectorCL (cl-real-vector settings width-bytes n)]
      (enq-fill! (.queue settings) (.cl-buf zero-sv) zero-array)
      zero-sv))
  (byte-size [_]
    width-bytes)
  (copy [_ y]
    (do (enq-copy! cl-buf (.cl-buf ^RealVectorCL y))
        y))
  (swp [x y]
    (do
      (s-swp engine cl-buf (.cl-buf ^RealVectorCL y))
      x)))

(defn cl-real-vector
  ([^CLSettings settings cl-buf ^long width-bytes ^long dim]
   (->RealVectorCL settings
                   (gcn-vector-engine (.context settings)
                                      (.queue settings)
                                      (.prog settings)
                                      (.max-local-size settings)
                                      cl-buf
                                      width-bytes
                                      dim)
                   cl-buf
                   width-bytes
                   dim))

  ([^CLSettings settings ^long width-bytes ^long dim]
   (cl-real-vector settings
                (cl-buffer (.context settings) (* dim width-bytes) :read-write)
                width-bytes dim)))

(defn cl-sv [settings ^long dim]
  (cl-real-vector settings Float/BYTES dim))

;; ======================= Dense Matrix ========================================
