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
             [utils :refer [dragan-says-ex]]]
            [uncomplicate.neanderthal
             [core :refer [dim transfer transfer! subvector matrix?]]
             [integer :refer [amax]]]
            [uncomplicate.neanderthal.internal
             [api :refer :all :exclude [amax]]
             [navigation :refer :all]]
            [uncomplicate.neanderthal.internal.cpp.structures :refer [cs-vector]]
            [uncomplicate.neanderthal.internal.cpp.mkl.structures :refer [seq-to-csr]])
  (:import [uncomplicate.neanderthal.internal.cpp.structures CSVector]))

(defn csv?
  "TODO"
  [x]
  (instance? CSVector x))

(defn csv
  "TODO"
  ([fact n idx nz & nzs]
   (let [fact (factory fact)
         idx-fact (index-factory fact)
         idx (or idx [])
         nz (or nz [])]
     (let-release [idx (if (compatible? idx-fact idx)
                         (view idx)
                         (transfer idx-fact idx))
                   res (cs-vector fact n (view idx) true)]
       (if-not nzs
         (transfer! nz (entries res))
         (transfer! (cons nz nzs) (entries res)))
       res)))
  ([fact ^long n source]
   (if (csv? source)
     (csv fact n (indices source) (entries source))
     (if (number? (first source))
       (csv fact n source nil)
       (csv fact n (first source) (second source)))))
  ([fact cs]
   (if (number? cs)
     (csv fact cs nil)
     (csv fact (dim cs) (indices cs) (entries cs)))))

(defn csr?
  "TODO"
  [x]
  (satisfies? CSR x))

(defn csr
  "TODO"
  ([fact m n idx idx-b idx-e nz options] ;; TODO error messages
   (if (and (<= 0 (long m)) (<= 0 (long n)))
     (let [idx-fact (index-factory fact)
           column? (= :column (:layout options))]
       (let-release [idx (if (compatible? idx-fact idx)
                           (view idx)
                           (transfer idx-fact idx))
                     idx-b (if (compatible? idx-fact idx-b)
                             (view idx-b)
                             (transfer idx-fact idx-b))
                     idx-e (if (compatible? idx-fact idx-e)
                             (view idx-e)
                             (transfer idx-fact idx-e))
                     res (create-ge-csr (factory fact) m n idx idx-b idx-e
                                        column? true)]
         (when-not (and (< (amax idx) (max 1 (long (if column? m n))))
                        (or (zero? (dim idx)) (< (amax idx-b) (dim idx)))
                        (<= (amax idx-e) (dim idx)))
           (dragan-says-ex "Sparse index outside of bounds."
                           {:requested (amax idx) :available (dim (entries res))}))
         (when nz (transfer! nz (entries res)))
         res))
     (dragan-says-ex "Compressed sparse matrix cannot have a negative dimension." {:m m :n n})))
  ([fact m n idx idx-be nz options]
   (csr fact m n idx (pop idx-be) (rest idx-be) nz options))
  ([fact m n idx idx-be options]
   (csr fact m n idx (pop idx-be) (rest idx-be) nil options))
  ([fact m n source options]
   (if (csr? source)
     (csr fact m n (columns source) (indexb source) (indexe source) (entries source) options)
     (if (nil? source)
       (let-release [idx-be (repeat (if (= :column (:layout options)) n m) 0)]
         (csr fact m n [] idx-be idx-be [] options))
       (let [[_ ptrs idx vals] (seq-to-csr source)]
         (csr fact m n idx (pop ptrs) (rest ptrs) vals options)))))
  ([fact m n arg]
   (if (map? arg)
     (csr fact m n nil arg)
     (csr fact m n arg nil)))
  ([arg0 arg1 arg2]
   (if (csr? arg0)
     (create-ge-csr (factory arg0) arg0 arg1 (= true (:index arg2)))
     (csr arg0 arg1 arg2 nil nil)))
  ([fact a]
   (let-release [res (transfer (factory fact) a)]
     (if (csr? res)
       res
       (dragan-says-ex "This is not a valid source for CSR matrices.")))))
