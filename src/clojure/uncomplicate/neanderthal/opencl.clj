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
           [uncomplicate.neanderthal.cblas FloatBlockVector]
           [java.nio ByteBuffer]))

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
  (let [prog (build-program! (program-with-source [(slurp "resources/opencl/blas.cl")]))
        max-local-size (max-work-group-size (queue-device queue))]
    (->CLSettings max-local-size queue (queue-context queue) prog)))

(def zero-array (float-array [0]))

(deftype FloatCLBlockVector [^CLSettings settings cl-buf ^long n
                             linear-work-size swp-kernel]
  RealVector
  (dim [_]
    n)
  (entry [_ i]
    (throw (UnsupportedOperationException.)))
  (subvector [_ k l]
    (cl-sv* settings
            (cl-sub-buffer cl-buf (* Float/BYTES k) (* Float/BYTES l))
            l))
  Releaseable
  (release [_]
    (and
     (release cl-buf)
     (release swp-kernel)))
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
    (do (set-arg! swp-kernel 1 (.cl-buf ^FloatCLBlockVector y))
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
                           (doto (kernel prog "swp_kernel") (set-arg! 0 cl-buf))))))

(defn cl-sv [settings ^long dim]
  (cl-sv* settings dim))
