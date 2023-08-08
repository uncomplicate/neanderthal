;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.constants)

(def ^:const blas-layout
  {:row 101
   :column 102
   :col 102})

(def ^:const blas-uplo
  {:upper 121
   :lower 122
   :up 121
   :low 122})

(def ^:const blas-transpose
  {:no-trans 111
   :trans 112})

(def ^:const blas-side
  {:left 141
   :right 142})

(def ^:const blas-diag
  {:unit 131
   :non-unit 132})
