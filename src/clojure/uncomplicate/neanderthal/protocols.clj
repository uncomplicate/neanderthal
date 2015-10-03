(ns uncomplicate.neanderthal.protocols)

(defprotocol Factory
  (create-vector1 [this n source])
  (create-matrix [this m n source])
  (data-accessor [this])
  (vector-engine [this buf n ofst strd])
  (matrix-engine [this buf m n ofst ld]))

(defprotocol EngineProvider
  (engine ^BLAS [this]))

(defprotocol CLAccessor
  (get-queue [this])
  (fill-buffer [this cl-buf val])
  (wrap-seq [this s])
  (wrap-prim [this p]))

(defprotocol BlockCreator
  (create-block [this n][this m n]))

(defprotocol Memory
  (compatible [this other]))

(defprotocol Mappable
  (map-memory [this] [this flags])
  (unmap [this mapped]))

(defprotocol Group
  (zero [this]))

(defprotocol Functor
  (fmap! [x f] [x f y] [x f y z]
    [x f y z v] [x f y z v ws]))

(defprotocol Foldable
  (fold [x] [x f] [x f acc]))

(defprotocol Reducible
  (freduce [x f] [x acc f] [x acc f y]
    [x acc f y z] [x acc f y z ws]))
