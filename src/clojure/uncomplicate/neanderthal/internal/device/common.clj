;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.device.common
  (:require [uncomplicate.neanderthal.internal.api :refer [COLUMN_MAJOR LOWER UPPER]])
  (:import [uncomplicate.neanderthal.internal.api ContiguousBlock TRMatrix]))

(defn name-transp [name ^ContiguousBlock a ^ContiguousBlock b]
  (format "%s_%s" name (if (= (.order a) (.order b)) "no_transp" "transp")))

(defn tr-bottom [^TRMatrix a]
  (if (= (.order a) COLUMN_MAJOR)
    (= (.uplo a) LOWER)
    (= (.uplo a) UPPER)))

(defn fits-buffer? [^ContiguousBlock a ^ContiguousBlock b]
  (and (= (.order a) (.order b)) (= (.sd a) (.sd b) (.stride a) (.stride b))))
