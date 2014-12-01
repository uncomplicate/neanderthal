(ns uncomplicate.neanderthal.protocols)

(def ^:const ROW_MAJOR 101)
(def ^:const COLUMN_MAJOR 102)
(def ^:const NO_TRANS 111)
(def ^:const TRANS 112)

(defprotocol Carrier
  (zero [_]))

(defprotocol Functor
  (fmap! [x f] [x f y] [x f y z]
    [x f y z v] [x f y z v ws]))

(defprotocol Foldable
  (fold [x] [x f] [x f acc]))

(defprotocol Reducible
  (freduce [x f] [x acc f] [x acc f y]
    [x acc f y z] [x acc f y z  ws]))
