(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.neanderthal
             [protocols :as p]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [max-work-group-size queue-device queue-context]]])
  (:import [uncomplicate.neanderthal.protocols
            RealVector RealMatrix Carrier
            RealChangeable]
           [uncomplicate.clojurecl.core WorkSize]
           [uncomplicate.neanderthal.cblas FloatBlockVector]
           [java.nio ByteBuffer]))

(defn power-of-2? [^long n]
  (= 0 (bit-and n (- n 1))))

(defn reduction-work-sizes [^long max-local-size ^long n]
  (if (power-of-2? n)
    (loop [acc [] global-size n]
      (if (= 1 global-size)
        acc
        (let [local-size (min global-size max-local-size)]
          (recur (conj acc (work-size [global-size] [local-size]))
                 (quot (+ global-size (dec local-size)) local-size)))))
    (throw (UnsupportedOperationException.
            (format "Reduction is supported in CL only on power-of-2 number of elements: %d." n)))))

(defn doreduction [queue reduction-kernel work-sizes cl-buf]
  (let [res (float-array 1)]
    (set-arg! reduction-kernel 0 cl-buf)
    (doseq [wsize work-sizes]
      (set-arg! reduction-kernel 1 (* Float/BYTES (aget ^longs (.local ^WorkSize wsize) 0)))
      (enq-nd! queue reduction-kernel wsize))
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
        prog (build-program! (program-with-source [(slurp "resources/opencl/blas.cl")]))
        max-local-size (max-work-group-size dev)]
    (->CLSettings max-local-size queue (queue-context queue) prog)))

(def zero-array (float-array [0]))

(deftype FloatCLBlockVector [^CLSettings settings cl-buf ^long n
                             linear-work-size swp-kernel scal-kernel
                             axpy-kernel
                             dot-local-reduce-kernel
                             sum-reduction-kernel]

  Releaseable
  (release [_]
    (and
     (release cl-buf)
     (release swp-kernel)
     (release scal-kernel)
     (release axpy-kernel)
     (release dot-local-reduce-kernel)
     (release sum-reduction-kernel)))

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
    (let [group-size (aget ^longs (.local ^WorkSize linear-work-size) 0)
          res-count (long (quot (+ n (dec group-size)) group-size))]
      (if (power-of-2? n)
        (with-release [res (cl-buffer (.context settings) (* Float/BYTES res-count) :read-write)]
          (set-arg! dot-local-reduce-kernel 1 (.cl-buf ^FloatCLBlockVector y))
          (set-arg! dot-local-reduce-kernel 2 res)
          (set-arg! dot-local-reduce-kernel 3 (* Float/BYTES group-size))
          (enq-nd! (.queue settings) dot-local-reduce-kernel linear-work-size)
          (doreduction (.queue settings) sum-reduction-kernel
                       (reduction-work-sizes (.max-local-size settings) res-count)
                       res))
        (throw (UnsupportedOperationException.
                (format "Dot is supported only on CL vectors with power-of-2 dimension: %d." n))))))
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
                           (doto (kernel prog "dot_local_reduce") (set-arg! 0 cl-buf))
                           (kernel prog "local_reduce_destructive")))))

(defn cl-sv [settings ^long dim]
  (cl-sv* settings dim))
