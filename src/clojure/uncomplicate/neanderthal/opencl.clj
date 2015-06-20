(ns ^{:author "Dragan Djuric"}
  uncomplicate.neanderthal.opencl
  (:require [uncomplicate.neanderthal
             [protocols :as p]]
            [uncomplicate.clojurecl
             [core :refer :all]
             [info :refer [max-work-group-size queue-device]]])
  (:import [uncomplicate.clojurecl.core CLBuffer]
           [uncomplicate.neanderthal.protocols
            RealVector RealMatrix Carrier
            RealChangeable]
           [uncomplicate.neanderthal.cblas FloatBlockVector]
           [java.nio ByteBuffer]))

(defprotocol Mappable
  (read! [this v])
  (write! [this v])
  (map-host [this])
  (unmap [this]))

(deftype FloatCLVector [cl-buf linear-work-size queue swp-kernel]
  Releaseable
  (release [_]
    (and
      (release cl-buf)
      (release swp-kernel)))
  Mappable
  (read! [_ host]
    (enq-read! queue cl-buf (.buf ^FloatBlockVector host)))
  (write! [_ host]
    (enq-write! queue cl-buf (.buf ^FloatBlockVector host)))
  Carrier
  (copy [_ y]
    (do (enq-copy! cl-buf (.cl-buf ^FloatCLVector y))
        y))
  (swp [x y]
    (do (set-args! swp-kernel cl-buf (.cl-buf ^FloatCLVector y))
        (enq-nd! queue swp-kernel linear-work-size)
        x)))

(defrecord CLSettings [^long max-local-size queue prog]
  Releaseable
  (release [_]
    (release prog)))

(defn cl-settings [queue]
  (let [prog (build-program! (program-with-source [(slurp "resources/opencl/blas.cl")]))
        max-local-size (max-work-group-size (queue-device queue))]
    (->CLSettings max-local-size queue prog)))

(defn cl-fv [^CLSettings settings ^long size]
  (let [cl-buf (cl-buffer (* size Float/BYTES))
        prog (.prog settings)]
    (->FloatCLVector cl-buf
                     (work-size [size] [(min size (.max-local-size settings))])
                     (.queue settings)
                     (kernel prog "swp_kernel"))))
