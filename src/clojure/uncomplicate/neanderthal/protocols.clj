(ns uncomplicate.neanderthal.protocols)

(defprotocol Carrier
  (column-major? [x]));; TODO check when this is needed, and possibly move it to another interface

(defprotocol Group
  (zero [_]))

(defprotocol Functor
  (fmap! [x f] [x f y] [x f y z]
    [x f y z v] [x f y z v ws]))

(defprotocol Foldable
  (fold [x] [x f] [x f acc]))

(defprotocol Reducible
  (freduce [x f] [x acc f] [x acc f y]
    [x acc f y z] [x acc f y z ws]))
