;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns ^{:author "Dragan Djuric"}
    uncomplicate.neanderthal.internal.device.random123
  (:require [uncomplicate.commons
             [core :refer [release]]
             [utils :refer [dragan-says-ex delete create-temp-dir]]])
  (:import [java.nio.file Files Path CopyOption FileVisitOption]
           java.nio.file.attribute.FileAttribute))

(defn copy-philox [^Path path]
  (let [random123-path (.resolve path "Random123")
        attributes (make-array FileAttribute 0)
        options (make-array CopyOption 0)]
    (try
      (Files/createDirectories (.resolve random123-path "features/dummy") attributes)
      (doseq [include-name ["philox.h" "array.h" "features/compilerfeatures.h"
                            "features/openclfeatures.h"]]
        (Files/copy
         (ClassLoader/getSystemResourceAsStream
          (format "uncomplicate/neanderthal/internal/device/include/Random123/%s" include-name))
         (.resolve random123-path ^String include-name)
         ^"[Ljava.nio.file.CopyOption;" options))
      (catch Exception e
        (delete path)
        (throw e)))))

(defmacro with-philox [path & body]
  `(try
     (copy-philox ~path)
     (do ~@body)
     (finally
       (delete ~path))))

(defn delete-shutdown [path]
  (.addShutdownHook (Runtime/getRuntime) (Thread. #(delete path))))

(defn release-deref [ds]
  (if (sequential? ds)
    (doseq [d ds]
      (when (realized? d) (release @d)))
    (when (realized? ds) (release ds))))

(defonce temp-dir (doto (create-temp-dir "uncomplicate_")
                    (copy-philox)
                    (delete-shutdown)))
