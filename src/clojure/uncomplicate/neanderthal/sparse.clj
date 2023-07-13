;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.sparse
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release view]]
             [utils :refer [dragan-says-ex direct-buffer]]]
            [uncomplicate.neanderthal
             [core :refer [dim transfer transfer! subvector matrix?]]
             [integer :refer [amax]]]
            [uncomplicate.neanderthal.internal
             [api :as api]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.cpp.structures
             :refer [cs-vector CompressedSparse indices entries CSR columns indexb indexe
                     create-ge-csr]]
            [uncomplicate.neanderthal.internal.cpp.mkl.structures
             :refer [seq-to-csr]])
  (:import [uncomplicate.neanderthal.internal.cpp.structures CSVector]))

(defn csv?
  "TODO"
  [x]
  (instance? CSVector x))

(defn csv
  "TODO"
  ([factory n idx nz & nzs]
   (let [factory (api/factory factory)
         idx-factory (api/index-factory factory)
         idx (or idx [])
         nz (or nz [])]
     (let-release [idx (if (api/compatible? idx-factory idx)
                         (view idx)
                         (transfer idx-factory idx))
                   res (cs-vector factory n (view idx) true)]
       (if-not nzs
         (transfer! nz (entries res))
         (transfer! (cons nz nzs) (entries res)))
       res)))
  ([factory ^long n source]
   (if (csv? source)
     (csv factory n (indices source) (entries source))
     (if (number? (first source))
       (csv factory n source nil)
       (csv factory n (first source) (second source)))))
  ([factory cs]
   (if (number? cs)
     (csv factory cs nil)
     (csv factory (dim cs) (indices cs) (entries cs)))))

(defn csr?
  "TODO"
  [x]
  (satisfies? CSR x))

(defn csr
  "TODO"
  ([factory m n idx idx-b idx-e nz options] ;; TODO error messages
   (if (and (<= 0 (long m)) (<= 0 (long n)))
     (let [idx-factory (api/index-factory factory)
           column? (= :column (:layout options))]
       (let-release [idx (if (api/compatible? idx-factory idx)
                           (view idx)
                           (transfer idx-factory idx))
                     idx-b (if (api/compatible? idx-factory idx-b)
                             (view idx-b)
                             (transfer idx-factory idx-b))
                     idx-e (if (api/compatible? idx-factory idx-e)
                             (view idx-e)
                             (transfer idx-factory idx-e))
                     res (create-ge-csr (api/factory factory) m n idx idx-b idx-e
                                        column? true)]
         (when-not (and (< (long (amax idx)) (max 1 (long (if column? m n))))
                        (or (zero? (dim idx)) (< (long (amax idx-b)) (dim idx)))
                        (<= (long (amax idx-e)) (dim idx)))
           (dragan-says-ex "Sparse index outside of bounds."
                           {:requested (amax idx) :available (dim (entries res))}))
         (when nz (transfer! nz (entries res)))
         res))
     (dragan-says-ex "Compressed sparse matrix cannot have a negative dimension." {:m m :n n})))
  ([factory m n idx idx-be nz options]
   (csr factory m n idx (pop idx-be) (rest idx-be) nz options))
  ([factory m n idx idx-be options]
   (csr factory m n idx (pop idx-be) (rest idx-be) nil options))
  ([factory m n source options]
   (if (csr? source)
     (csr factory m n (columns source) (indexb source) (indexe source) (entries source) options)
     (if (nil? source)
       (let-release [idx-be (repeat (if (= :column (:layout options)) n m) 0)]
         (csr factory m n [] idx-be idx-be [] options))
       (let [[_ ptrs idx vals] (seq-to-csr source)]
         (csr factory m n idx (pop ptrs) (rest ptrs) vals options)))))
  ([factory m n arg]
   (if (map? arg)
     (csr factory m n nil arg)
     (csr factory m n arg nil)))
  ([arg0 arg1 arg2]
   (if (csr? arg0)
     (create-ge-csr (api/factory arg0) arg0 arg1 (= true (:index arg2)))
     (csr arg0 arg1 arg2 nil nil)))
  ([factory a]
   (let-release [res (transfer (api/factory factory) a)]
     (if (csr? res)
       res
       (dragan-says-ex "This is not a valid source for CSR matrices.")))))
