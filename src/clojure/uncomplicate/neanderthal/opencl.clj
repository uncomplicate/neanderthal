(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.neanderthal.math :refer [power-of-2?]]
   [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [max-work-group-size queue-device queue-context]]])
  (:import [uncomplicate.neanderthal.protocols
            RealVector RealMatrix Carrier
            RealChangeable]
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

(defn ^:private enq-reduce! [queue reduction-kernel work-sizes cl-buf]
  (let [acc (float-array 1)]
    (set-arg! reduction-kernel 0 cl-buf)
    (doseq [wsize work-sizes]
      (set-arg! reduction-kernel 1 (* Float/BYTES (aget ^longs (.local ^WorkSize wsize) 0)))
      (enq-nd! queue reduction-kernel wsize))
    (enq-read! queue cl-buf acc)
    (aget acc 0)))

(defn ^:private enq-reduce [context queue main-kernel reduce-kernel
                            max-local-size n]
  (if (power-of-2? n)
    (let [local-size (min (long n) (long max-local-size))
          acc-count (long (quot (+ (long n) (dec local-size)) local-size))]
      (with-release [acc (cl-buffer context (* Float/BYTES acc-count) :read-write)]
        (set-arg! main-kernel 0 acc)
        (set-arg! main-kernel 1 (* Float/BYTES local-size))
        (enq-nd! queue main-kernel (work-size [n] [local-size]))
        (enq-reduce! queue reduce-kernel
                     (reduction-work-sizes max-local-size acc-count)
                     acc)))
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
        prog (build-program! (program-with-source [(slurp "resources/opencl/blas.cl")]))
        max-local-size (max-work-group-size dev)]
    (->CLSettings max-local-size queue (queue-context queue) prog)))

(def zero-array (float-array [0]))

(deftype FloatCLBlockVector [^CLSettings settings cl-buf ^long n
                             linear-work-size swp-kernel scal-kernel
                             axpy-kernel
                             reduce-kernel
                             dot-reduce-kernel]

  Releaseable
  (release [_]
    (and
     (release cl-buf)
     (release swp-kernel)
     (release scal-kernel)
     (release axpy-kernel)
     (release reduce-kernel)
     (release dot-reduce-kernel)))

  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (throw (UnsupportedOperationException.)))
  (subvector [_ k l]
    (cl-sv* settings
            (cl-sub-buffer cl-buf (* Float/BYTES k) (* Float/BYTES l))
            l))
  (dot [_ y]
    (do
      (set-arg! dot-reduce-kernel 3 (.cl-buf ^FloatCLBlockVector y))
      (enq-reduce (.context settings) (.queue settings)
                  dot-reduce-kernel reduce-kernel
                  (.max-local-size settings) n)))
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
   (let [prog (.prog settings)]
     (->FloatCLBlockVector settings
                           cl-buf
                           dim
                           (work-size [dim] [(min dim (.max-local-size settings))])
                           (doto (kernel prog "swp") (set-arg! 0 cl-buf))
                           (doto (kernel prog "scal") (set-arg! 1 cl-buf))
                           (doto (kernel prog "axpy") (set-arg! 1 cl-buf))
                           (kernel prog "reduce")
                           (doto (kernel prog "dot_reduce") (set-arg! 2 cl-buf))))))

(defn cl-sv [settings ^long dim]
  (cl-sv* settings dim))
