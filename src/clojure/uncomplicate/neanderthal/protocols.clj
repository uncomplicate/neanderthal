(ns uncomplicate.neanderthal.protocols)

(defprotocol Group
  (zero [this]))

(defprotocol EngineFactory
  (data-accessor [this])
  (vector-engine [this buf n])
  (matrix-engine [this buf m n]))

(defprotocol EngineProvider
  (engine ^BLAS [this]))

(defprotocol CLAccessor
  (get-queue [this])
  (create-buffer [this n])
  (fill-buffer [this cl-buf val])
  (array [this s])
  (slice [this cl-buf k l]))

(defprotocol BlockCreator
  (create-block [this n][this m n]))

(defprotocol Memory
  (compatible [this other]))

(defprotocol Mappable
  (read! [this host])
  (write! [this host])
  (map-host [this])
  (unmap [this]))

(defprotocol Functor
  (fmap! [x f] [x f y] [x f y z]
    [x f y z v] [x f y z v ws]))

(defprotocol Foldable
  (fold [x] [x f] [x f acc]))

(defprotocol Reducible
  (freduce [x f] [x acc f] [x acc f y]
    [x acc f y z] [x acc f y z ws]))
