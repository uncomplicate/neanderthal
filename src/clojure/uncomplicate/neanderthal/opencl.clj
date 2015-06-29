(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.neanderthal.math :refer [power-of-2?]]
   [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [max-work-group-size queue-device queue-context]]])
  (:import [uncomplicate.neanderthal.protocols
            Vector RealVector Matrix RealMatrix Carrier RealChangeable]
           [uncomplicate.clojurecl.core WorkSize]
           [uncomplicate.neanderthal.cblas FloatBlockVector]
           [java.nio ByteBuffer]))

(defn ^:private reduction-work-sizes [^long max-local-size ^long n]
  (loop [acc [] global-size n]
    (if (= 1 global-size)
      acc
      (let [local-size (min global-size max-local-size)]
        (recur (conj acc (work-size [global-size] [local-size]))
               (quot (+ global-size (dec local-size)) local-size))))))

(defn ^:private enq-reduce!
  ([queue reduction-kernel work-sizes cl-buf]
   (doseq [wsize work-sizes]
     (set-arg! reduction-kernel 0 (* Float/BYTES (aget ^longs (.local ^WorkSize wsize) 0)))
     (enq-nd! queue reduction-kernel wsize)))
  ([queue reduction-kernel work-sizes cl-buf-0 cl-buf-1]
   (do
     (set-arg! reduction-kernel 2 cl-buf-0)
     (set-arg! reduction-kernel 3 cl-buf-1)
     (doseq [wsize work-sizes]
       (set-args! reduction-kernel 0 (* Float/BYTES (aget ^longs (.local ^WorkSize wsize) 0))
              (* Integer/BYTES (aget ^longs (.local ^WorkSize wsize) 0)))
       (enq-nd! queue reduction-kernel wsize)))))

(defn ^:private enq-reduce [context queue main-kernel reduce-kernel
                            max-local-size n acc]
  (if (power-of-2? n)
    (let [res (float-array 1)
          local-size (min (long n) (long max-local-size))
          acc-count (long (quot (+ (long n) (dec local-size)) local-size))]
      (enq-nd! queue main-kernel (work-size [n] [local-size]))
      (enq-reduce! queue reduce-kernel
                   (reduction-work-sizes max-local-size acc-count)
                   acc)
      (enq-read! queue acc res)
      (aget res 0))
    (throw (UnsupportedOperationException.
            (format "Reduction is supported in CL only on power-of-2 number of elements: %d." n)))))

(defn ^:private enq-reduce2 [context queue main-kernel reduce-kernel
                            max-local-size n]
  (if (power-of-2? n)
    (let [res (int-array 1)
          entry-size Float/BYTES
          local-size (min (long n) (long max-local-size))
          acc-count (long (quot (+ (long n) (dec local-size)) local-size))]
      (with-release [acc-0 (cl-buffer context (* entry-size acc-count) :read-write)
                     acc-1 (cl-buffer context (* entry-size acc-count) :read-write)]
        (set-arg! main-kernel 0 (* entry-size local-size))
        (set-arg! main-kernel 1 (* entry-size local-size))
        (set-arg! main-kernel 2 acc-0)
        (set-arg! main-kernel 3 acc-1)
        (enq-nd! queue main-kernel (work-size [n] [local-size]))
        (enq-reduce! queue reduce-kernel
                     (reduction-work-sizes max-local-size acc-count)
                     acc-0 acc-1)
        (enq-read! queue acc-0 res)
        (aget res 0)))
    (throw (UnsupportedOperationException.
            (format "Reduction is supported in CL only on power-of-2 number of elements: %d." n)))))


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
        prog (build-program! (program-with-source [(slurp "resources/opencl/blas.cl")]))]
    (->CLSettings (max-work-group-size dev) queue (queue-context queue) prog)))

(def zero-array (float-array [0]))

(deftype FloatCLBlockVector [^CLSettings settings cl-buf ^long n
                             reduce-acc
                             linear-work-size
                             swp-kernel
                             scal-kernel
                             axpy-kernel
                             sum-reduction-kernel
                             max-reduction-kernel
                             dot-reduce-kernel
                             asum-reduce-kernel
                             iamax-reduce-kernel]

  Releaseable
  (release [_]
    (and
     (release cl-buf)
     (release reduce-acc)
     (release swp-kernel)
     (release scal-kernel)
     (release axpy-kernel)
     (release sum-reduction-kernel)
     (release max-reduction-kernel)
     (release dot-reduce-kernel)
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
    (enq-reduce2 (.context settings) (.queue settings)
                iamax-reduce-kernel max-reduction-kernel
                (.max-local-size settings) n))
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
            "To acces values from the CL device, read! them first.")))
  (dot [_ y]
    (do
      (set-arg! dot-reduce-kernel 3 (.cl-buf ^FloatCLBlockVector y))
      (enq-reduce (.context settings) (.queue settings)
                  dot-reduce-kernel sum-reduction-kernel
                  (.max-local-size settings) n reduce-acc)))
  (nrm2 [_]
    (throw (UnsupportedOperationException.
            "TODO.")))
  (asum [_n]
    (enq-reduce (.context settings) (.queue settings)
                asum-reduce-kernel sum-reduction-kernel
                (.max-local-size settings) n reduce-acc))
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
  (write! [this host]
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
         cl-acc (cl-buffer (.context settings) (* Float/BYTES acc-count) :read-write)];; TODO TODO this should be a part of the engine (queue)
     (->FloatCLBlockVector settings
                           cl-buf
                           dim
                           cl-acc
                           (work-size [dim] [(min dim (.max-local-size settings))])
                           (doto (kernel prog "swp") (set-arg! 0 cl-buf))
                           (doto (kernel prog "scal") (set-arg! 1 cl-buf))
                           (doto (kernel prog "axpy") (set-arg! 1 cl-buf))
                           (doto (kernel prog "sum_reduction") (set-arg! 1 cl-acc))
                           (kernel prog "max_reduction")
                           (doto (kernel prog "dot_reduce")
                             (set-args! 0 (* Float/BYTES local-size) cl-acc cl-buf))
                           (doto (kernel prog "asum_reduce")
                             (set-args! 0 (* Float/BYTES local-size) cl-acc cl-buf))
                           (doto (kernel prog "iamax_reduce") (set-arg! 4 cl-buf))
                           ))))

(defn cl-sv [settings ^long dim]
  (cl-sv* settings dim))

;; ======================= Dense Matrix ========================================
