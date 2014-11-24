(ns uncomplicate.neanderthal.protocols)

(def ^:const ROW_MAJOR 101)
(def ^:const COLUMN_MAJOR 102)
(def ^:const NO_TRANS 111)
(def ^:const TRANS 112)

(defprotocol Carrier
  (zero [_]))

