;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.mkl.core-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release release view]]
            [uncomplicate.clojure-cpp
             :refer [float-pointer int-pointer put! pointer-seq null? get! address get-entry capacity]]
            [uncomplicate.neanderthal.internal.cpp.mkl.core :refer :all])
  (:import  clojure.lang.ExceptionInfo
            [org.bytedeco.javacpp IntPointer FloatPointer]))

;; ================= Sparse tests ===============================



(facts "MKL export_csr test."
       (let [m 4
             n 3
             export-nz (float-pointer nil)
             export-pb (int-pointer nil)
             export-pe (int-pointer nil)
             export-indx (int-pointer nil)
             export-m (int-pointer 1)
             export-n (int-pointer 1)
             export-indexing (int-pointer 1)]
         (with-release [nz (float-pointer [10 15 20 30 40])
                        pb (int-pointer [0 2 3 4])
                        pe (int-pointer [2 3 4 5])
                        indx (int-pointer [2 3 2 1 0])
                        sm (create-csr nz 0 m n pb pe indx)
                        export-sm (sparse-matrix)]

           (pointer-seq nz) => [10.0 15.0 20.0 30.0 40.0]
           (null? export-nz) => true
           (null? export-pb) => true
           (export-csr export-nz sm export-indexing export-m export-n export-pb export-pe export-indx)
           (null? export-nz) => false
           (null? export-pb) => false
           (address export-nz) => (address nz)
           (address export-pb) => (address pb)
           (address export-pe) => (address pe)
           (address export-indx) => (address indx)
           (capacity export-nz) => 5
           (pointer-seq export-nz) => [10.0 15.0 20.0 30.0 40.0]
           (pointer-seq export-pb) => [0 2 3 4]
           (pointer-seq export-pe) => [2 3 4 5]
           (pointer-seq indx) => [2 3 2 1 0])))

(facts "MKL sp2m test."
       (let [m 4
             n 3
             export-nz (float-pointer nil)
             export-pb (int-pointer nil)
             export-pe (int-pointer nil)
             export-indx (int-pointer nil)
             export-m (int-pointer 1)
             export-n (int-pointer 1)
             export-indexing (int-pointer 1)]
         (with-release [nza (float-pointer [10 15 20 30 40])
                        pba (int-pointer [0 2 3 4])
                        pea (int-pointer [2 3 4 5])
                        indxa (int-pointer [2 3 2 1 0])
                        descr (matrix-descr :ge)
                        a (create-csr nza 0 m n pba pea indxa)
                        nzb (float-pointer [7 8 9 10])
                        pbb (int-pointer [0 2 3])
                        peb (int-pointer [2 3 4])
                        indxb (int-pointer [2 3 2 1])
                        b (create-csr nzb 0 n m pbb peb indxb)
                        c (sparse-matrix)]

           (sp2m 10 descr a 10 descr b :count c) => c
           (export-csr export-nz c export-indexing export-m export-n export-pb export-pe export-indx) => c
           (null? export-nz) => true
           (null? export-indx) => true
           (get-entry export-m) => 4
           (get-entry export-n) => 4
           (get-entry export-indexing) => 0
           (pointer-seq export-pb) => [0 1 2 3]
           (pointer-seq export-pe) => [1 2 3 5]

           (let [pe-address (address export-pe)]
             (sp2m 10 descr a 10 descr b :finalize-no-val c) => c
             (export-csr export-nz c export-indexing export-m export-n export-pb export-pe export-indx) => c
             (address export-pe) => pe-address)
           (pointer-seq export-indx) => [1 1 2 2 3]
           (null? export-nz) => true

           (let [pe-address (address export-pe)
                 indx-address (address export-indx)]
             (sp2m 10 descr a 10 descr b :finalize c) => c
             (export-csr export-nz c export-indexing export-m export-n export-pb export-pe export-indx) => c
             (address export-pe) => pe-address
             (address export-indx) => indx-address)
           (pointer-seq export-indx) => [1 1 2 2 3]
           (pointer-seq export-nz) => [100.0 200.0 270.0 280.0 320.0]

           (let [pe-address (address export-pe)]
             (sp2m 10 descr a 10 descr b :count c) => c
             (export-csr export-nz c export-indexing export-m export-n export-pb export-pe export-indx) => c
             (address export-pe) =not=> pe-address))))
