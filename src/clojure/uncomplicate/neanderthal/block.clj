;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.block
  (:require [uncomplicate.neanderthal.internal.api
             :refer [dec-property COLUMN_MAJOR ROW_MAJOR UPPER LOWER DIAG_UNIT DIAG_NON_UNIT]])
  (:import [uncomplicate.neanderthal.internal.api Block BlockMatrix DenseMatrix UPLOMatrix TRMatrix]))

(defn buffer [^Block x]
  (.buffer x))

(defn offset ^long [^Block x]
  (.offset x))

(defn stride ^long [^Block x]
  (.stride x))

(defn order [^DenseMatrix x]
  (dec-property (.order x)))

(defn sd ^long [^DenseMatrix b]
  (.sd b))

(defn fd ^long [^DenseMatrix b]
  (.fd b))

(defn block? [x]
  (instance? Block x))

(defn ecount
  "Returns the total number of elements in all dimensions of a block x
  of (possibly strided) memory.

  (ecount (dv 1 2 3)) => 3
  (ecount (dge 2 3)) => 6
  "
  ^long [^Block x]
  (.count x))

(defn column? [^BlockMatrix a]
  (= COLUMN_MAJOR (.order a)))

(defn row? [^BlockMatrix a]
  (= ROW_MAJOR (.order a)))

(defn upper? [^UPLOMatrix a]
  (= UPPER (.uplo a)))

(defn lower? [^UPLOMatrix a]
  (= LOWER (.uplo a)))

(defn unit-diag? [^TRMatrix a]
  (= DIAG_UNIT (.diag a)))

(defn non-unit-diag? [^TRMatrix a]
  (= DIAG_NON_UNIT (.diag a)))
