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

(defn ^:private enq-read-float ^double [queue cl-buf]
  (let [res (float-array 1)]
    (enq-read! queue cl-buf res)
    (aget res 0)))

(defprotocol Mappable
  (read! [this v])
  (write! [this v])
  (map-host [this])
  (unmap [this]))

(declare cl-sv*)

(defrecord CLSettings [^long max-local-size queue context prog]
  Releaseable
  (release [_]
    (release prog)))

(defn cl-settings [queue]
  (let [dev (queue-device queue)
        ctx (queue-context queue)
        prog (build-program! (program-with-source ctx [(slurp "resources/opencl/blas.cl")]) "-cl-std=CL2.0" nil)]
    (->CLSettings (max-work-group-size dev) queue ctx prog)))

(def zero-array (float-array [0]))

(deftype FloatCLBlockVector [^CLSettings settings cl-buf ^long n
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
     (release cl-buf)
     (release reduce-acc)
     (release reduce-iacc)
     (release swp-kernel)
     (release scal-kernel)n
     (release axpy-kernel)
     (release sum-reduction-kernel)
     (release imax-reduction-kernel)
     (release dot-reduce-kernel)
     (release nrm2-reduce-kernel)
     (release asum-reduce-kernel)
     (release iamax-reduce-kernel)))

  Vector
  (dim [_]
    n)
  (subvector [_ k l]
    (cl-sv* settings
            (cl-sub-buffer cl-buf (* Float/BYTES k) (* Float/BYTES l))
            l))
  (iamax [_]
    (do
      (enq-reduce (.queue settings)
                  iamax-reduce-kernel imax-reduction-kernel
                  (.max-local-size settings) n)
      (enq-read-int (.queue settings) reduce-iacc)))
  (rotg [x]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (rotm [x y p]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (rotmg [p args]
    (throw (UnsupportedOperationException.
            "TODO.")))
  RealVector
  (entry [_ i]
    (throw (UnsupportedOperationException.
            "To access values from the CL device, read! them first.")))
  (dot [_ y]
    (do
      (set-arg! dot-reduce-kernel 2 (.cl-buf ^FloatCLBlockVector y))
      (enq-reduce (.queue settings)
                  dot-reduce-kernel sum-reduction-kernel
                  (.max-local-size settings) n)
      (enq-read-float (.queue settings) reduce-acc)))
  (nrm2 [_]
    (do (enq-reduce (.queue settings)
                    nrm2-reduce-kernel sum-reduction-kernel
                    (.max-local-size settings) n)
        (sqrt (enq-read-float (.queue settings) reduce-acc))))
  (asum [_n]
    (do (enq-reduce (.queue settings)
                    asum-reduce-kernel sum-reduction-kernel
                    (.max-local-size settings) n)
        (enq-read-float (.queue settings) reduce-acc)))
  (rot [x y c s]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (scal [x alpha]
    (do
      (set-arg! scal-kernel 0 (float-array [alpha]))
      (enq-nd! (.queue settings) scal-kernel linear-work-size)
      x))
  (axpy [_ alpha y]
    (do
      (set-arg! axpy-kernel 0 (float-array [alpha]))
      (set-arg! axpy-kernel 2 (.cl-buf ^FloatCLBlockVector y))
      (enq-nd! (.queue settings) axpy-kernel linear-work-size)
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
  (write! [this host]n
    (do
      (enq-write! (.queue settings) cl-buf (.buf ^FloatBlockVector host))
      this))
  Carrier
  (zero [_]
    (let [zero-sv ^FloatCLBlockVector (cl-sv* settings n)]
      (enq-fill! (.queue settings) (.cl-buf zero-sv) zero-array)
      zero-sv))
  (byte-size [_]
    Float/BYTES)
  (copy [_ y]
    (do (enq-copy! cl-buf (.cl-buf ^FloatCLBlockVector y))
        y))
  (swp [x y]
    (do
      (set-arg! swp-kernel 1 (.cl-buf ^FloatCLBlockVector y))
      (enq-nd! (.queue settings) swp-kernel linear-work-size)
      x)))

(defn cl-sv*
  ([^CLSettings settings ^long dim]
   (cl-sv* settings
           (cl-buffer (.context settings) (* dim Float/BYTES) :read-write)
           dim))
  ([^CLSettings settings cl-buf ^long dim]
   (let [prog (.prog settings)
         local-size (min dim (long (.max-local-size settings)))
         acc-count (long (quot (+ dim (dec local-size)) local-size))
         ;; TODO TODO this should be a part of the engine (queue)
         cl-acc (cl-buffer (.context settings) (* Float/BYTES acc-count) :read-write)
         ;; TODO TODO this should be a part of the engine (queue)
         cl-iacc (cl-buffer (.context settings) (* Integer/BYTES acc-count) :read-write)]

     (->FloatCLBlockVector settings
                           cl-buf
                           dim
                           cl-acc
                           cl-iacc
                           (work-size [dim] [(min dim (.max-local-size settings))])
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
                           ))))

(defn cl-sv [settings ^long dim]
  (cl-sv* settings dim))

;; ======================= Dense Matrix ========================================
