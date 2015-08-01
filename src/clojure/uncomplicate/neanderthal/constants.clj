(ns uncomplicate.neanderthal.constants)

(def ^:const ROW_MAJOR 101)

(def ^:const COLUMN_MAJOR 102)

(def ^:const DEFAULT_ORDER COLUMN_MAJOR)

(def MAT_BOUNDS_MSG
  "Requested entry %d, %d is out of bounds of matrix %d x %d.")

(def INCOMPATIBLE_BLOCKS_MSG
  "Operation is not permited on vectors with incompatible buffers,
  or dimensions that are incompatible in the context of the operation.
  1: %s
  2: %s")
