;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.internal.cpp.mkl.core
  (:require [uncomplicate.commons
             [core :refer [with-release let-release Info info Releaseable release Viewable]]
             [utils :refer [dragan-says-ex with-check]]]
            [uncomplicate.clojure-cpp
             :refer [float-ptr double-ptr int-ptr int-ptr* null? get-entry capacity!]]
            [uncomplicate.neanderthal.internal.api :refer [Flippable]]
            [uncomplicate.neanderthal.internal.cpp.mkl.constants :refer :all])
  (:import [org.bytedeco.javacpp Pointer FloatPointer DoublePointer IntPointer]
           [org.bytedeco.mkl.global mkl_rt mkl_rt$MKLVersion mkl_rt$sparse_matrix mkl_rt$matrix_descr]))

;; ===================== System ================================================================

(defn version []
  (with-release [result (mkl_rt$MKLVersion.)]
    (mkl_rt/MKL_Get_Version result)
    {:major (.MajorVersion result)
     :minor (.MinorVersion result)
     :update (.UpdateVersion result)}))

;; ===================== Memory Management =====================================================

(defn malloc!
  ([^long size]
   (mkl_rt/MKL_malloc size -1))
  ([^long size ^long align]
   (mkl_rt/MKL_malloc size align)))

(defn calloc!
  ([^long n ^long element-size]
   (mkl_rt/MKL_calloc n element-size -1))
  ([^long n ^long element-size ^long align]
   (mkl_rt/MKL_calloc n element-size align)))

(defn realloc! [^Pointer p ^long size]
  (mkl_rt/MKL_realloc p size))

(defn free! [^Pointer p]
  (mkl_rt/MKL_free p)
  (.setNull p)
  p)

(defn free-buffers! []
  (mkl_rt/MKL_Free_Buffers))

(defn thread-free-buffers! []
  (mkl_rt/MKL_Thread_Free_Buffers))

(defn allocated-bytes ^long []
  (mkl_rt/MKL_Mem_Stat (int-array 3)))

(defn peak-mem-usage! ^long [mode]
  (let [result (mkl_rt/MKL_Peak_Mem_Usage (get mkl-peak-mem mode mode))]
    (if (= -1 result)
      (dragan-says-ex "MKL reported an unspecified error during this call.")
      result)))

(defn peak-mem-usage ^long []
  (peak-mem-usage! :report))

;; ===================== Miscellaneous =========================================================

(defn enable-instructions! [instruction-set]
  (dec-mkl-result (mkl_rt/MKL_Enable_Instructions (get mkl-enable-instructions instruction-set instruction-set))))

(defn verbose! [timing]
  (dec-verbose-mode (mkl_rt/MKL_Verbose (get mkl-verbose-mode timing timing))))

;; ===================== Sparse Matrix =========================================================

(defn mkl-sparse [type name]
  (symbol (format "mkl_sparse_%s_%s" type name)))

(defn sparse-error [^long err-code details]
  (let [err (dec-sparse-status err-code)]
    (ex-info (format "MKL sparse error: %s." (name err))
             {:name err :code err-code :type :sparse-error :details details})))

(extend-type mkl_rt$sparse_matrix
  Releaseable
  (release [this]
    (when-not (null? this)
      (with-check sparse-error
        (mkl_rt/mkl_sparse_destroy this)
        (do (.deallocate this)
            (.setNull this)
            true)))))

(defn sparse-matrix
  (^mkl_rt$sparse_matrix []
   (mkl_rt$sparse_matrix.))
  (^mkl_rt$sparse_matrix [m]
   m))

(defprotocol SparseRoutines
  (create-csr [nz indexing rows cols rows-start rows-end col-indx])
  (create-csc [nz indexing rows cols cols-start cols-end row-indx])
  (create-coo [nz indexing rows cols nnz row-indx col-indx])
  (export-csr [nz source indexing rows cols rows-start rows-end col-indx])
  (update-internal [nz n sparse-matrix]))

(defmacro create-cs [method indexing rows cols start end indx nz]
  `(let-release [sparse-matrix# (mkl_rt$sparse_matrix.)]
     (with-check sparse-error
       (. mkl_rt ~method sparse-matrix#
          (int ~indexing) (int ~rows) (int ~cols) (int-ptr ~start) (int-ptr ~end) (int-ptr ~indx) ~nz)
       sparse-matrix#)))

(defmacro export-cs [method source indexing rows cols start end indx nz]
  `(let [indexing# (int-ptr* ~indexing)
         rows# (int-ptr* ~rows)
         cols# (int-ptr* ~cols)
         start# (int-ptr* ~start)
         end# (int-ptr* ~end)
         indx# (int-ptr* ~indx)]
     (with-check sparse-error
       (. mkl_rt ~method (sparse-matrix ~source) indexing# rows# cols# start# end# indx# ~nz)
       (let [m# (long (get-entry rows# 0))
             nnz# (- (long (get-entry end# (dec m#))) (long (get-entry start# 0)))]
         (capacity! start# m#)
         (capacity! end# m#)
         (capacity! indx# nnz#)
         (capacity! ~nz nnz#)
         ~source))))

(defmacro extend-sparse-pointer [name t ptr]
  `(extend-type ~name
     SparseRoutines
     (create-csr [nz# indexing# rows# cols# rows-start# rows-end# col-indx#]
       (create-cs ~(mkl-sparse t 'create_csr)
                  indexing# rows# cols# rows-start# rows-end# col-indx# (~ptr nz#)))
     (create-csc [nz# indexing# rows# cols# cols-start# cols-end# row-indx#]
       (create-cs ~(mkl-sparse t 'create_csr)
                  indexing# rows# cols# cols-start# cols-end# row-indx# (~ptr nz#)))
     (create-coo [nz# indexing# rows# cols# nnz# row-indx# col-indx#]
       (let-release [sparse-matrix# (mkl_rt$sparse_matrix.)]
         (with-check sparse-error
           (. mkl_rt ~(mkl-sparse t 'create_coo) sparse-matrix# (int indexing#)
              (int rows#) (int cols#) (int nnz#) (int-ptr row-indx#) (int-ptr col-indx#) (~ptr nz#))
           sparse-matrix#)))
     (export-csr [nz# source# indexing# rows# cols# rows-start# rows-end# col-indx#]
       (export-cs ~(mkl-sparse t 'export_csr) source#
                  indexing# rows# cols# rows-start# rows-end# col-indx# nz#))
     (export-csc [nz# source# indexing# rows# cols# rows-start# rows-end# col-indx#]
       (export-cs ~(mkl-sparse t 'export_csc) source#
                  indexing# rows# cols# rows-start# rows-end# col-indx# nz#))
     (update-internal [nz# n# sparse-matrix#]
       (with-check sparse-error
         (. mkl_rt ~(mkl-sparse t 'update_values) (sparse-matrix sparse-matrix#) (int n#) nil nil nz#)
         sparse-matrix#))))

(defmacro extend-sparse-primitive [name t ptr]
  `(extend-type ~name
     SparseRoutines
     (set-internal [v# i# j#]
       (with-check sparse-error
         (. mkl_rt ~(mkl-sparse t 'update_values) (sparse-matrix sparse-matrix#) (int n#) nil nil nz#)
         sparse-matrix#))))

(extend-sparse-pointer DoublePointer "d" double-ptr)
(extend-sparse-pointer FloatPointer "s" float-ptr)

(defn matrix-descr*
  ([]
   (mkl_rt$matrix_descr.))
  ([^long type ^long mode ^long diag]
   (let-release [res (mkl_rt$matrix_descr.)]
     (doto res
       (.diag diag)
       (.mode mode)
       (.type type))))
  ([^mkl_rt$matrix_descr desc]
   (matrix-descr* (.type desc) (.mode desc) (.diag desc))))

(defn matrix-descr
  ([])
  ([arg]
   (if (instance? mkl_rt$matrix_descr arg)
     (matrix-descr* arg)
     (matrix-descr arg arg arg)))
  ([type arg]
   (matrix-descr type arg arg ))
  ([type mode diag]
   (matrix-descr* (get mkl-sparse-matrix-type type type) (get mkl-sparse-fill-mode mode mode)
                  (get mkl-sparse-diag-mode diag diag))))

(extend-type mkl_rt$matrix_descr
  Viewable
  (view [this]
    (matrix-descr this))
  Flippable
  (flip [this]
    (case (.mode this)
      40 (.mode this 41)
      41 (.mode this 40)
      this)))

(let [descr (matrix-descr :ge)]
  (defn mkl-sparse-copy [^mkl_rt$sparse_matrix source ^mkl_rt$sparse_matrix dest]
    (with-check sparse-error
      (mkl_rt/mkl_sparse_copy source descr dest)
      dest)))

(defn mkl-sparse-convert-csr [^mkl_rt$sparse_matrix source operation ^mkl_rt$sparse_matrix dest]
  (with-check sparse-error
    (mkl_rt/mkl_sparse_convert_csr source (get mkl-sparse-operation operation operation) dest)
    dest))

(defn sparse-mm-hint!
  ([^mkl_rt$sparse_matrix a ^mkl_rt$matrix_descr descr operation layout dense-size expected-calls]
   (with-check sparse-error
     (mkl_rt/mkl_sparse_set_mm_hint a (get mkl-sparse-operation operation operation)
                                    descr (get mkl-sparse-layout layout layout)
                                    dense-size expected-calls)
     a))
  ([desc operation layout dense-size expected-calls]
   (fn [a]
     (sparse-mm-hint! a desc operation layout dense-size expected-calls)))
  ([operation layout dense-size expected-calls]
   (fn [a desc]
     (sparse-mm-hint! a desc operation layout dense-size expected-calls))))

(defn sparse-mv-hint!
  ([^mkl_rt$sparse_matrix a ^mkl_rt$matrix_descr descr operation
    ^long expected-calls]
   (with-check sparse-error
     (mkl_rt/mkl_sparse_set_mv_hint a (get mkl-sparse-operation operation operation)
                                    descr expected-calls)
     a))
  ([desc operation expected-calls]
   (fn [a]
     (sparse-mv-hint! a desc operation expected-calls)))
  ([operation expected-calls]
   (fn [a desc]
     (sparse-mv-hint! a desc operation expected-calls))))

(defn sparse-dotmv-hint!
  ([^mkl_rt$sparse_matrix a ^mkl_rt$matrix_descr descr operation
    ^long expected-calls]
   (with-check sparse-error
     (mkl_rt/mkl_sparse_set_dotmv_hint a (get mkl-sparse-operation operation operation)
                                       descr expected-calls)
     a))
  ([desc operation expected-calls]
   (fn [a]
     (sparse-dotmv-hint! a desc operation expected-calls)))
  ([operation expected-calls]
   (fn [a desc]
     (sparse-dotmv-hint! a desc operation expected-calls))))

(defn sparse-symgs-hint!
  ([^mkl_rt$sparse_matrix a ^mkl_rt$matrix_descr descr operation
    ^long expected-calls]
   (with-check sparse-error
     (mkl_rt/mkl_sparse_set_symgs_hint a (get mkl-sparse-operation operation operation)
                                       descr expected-calls)
     a))
  ([desc operation expected-calls]
   (fn [a]
     (sparse-symgs-hint! a desc operation expected-calls)))
  ([operation expected-calls]
   (fn [a desc]
     (sparse-symgs-hint! a desc operation expected-calls))))

(defn sparse-sv-hint!
  ([^mkl_rt$sparse_matrix a ^mkl_rt$matrix_descr descr operation
    ^long expected-calls]
   (with-check sparse-error
     (mkl_rt/mkl_sparse_set_sv_hint a (get mkl-sparse-operation operation operation)
                                    descr expected-calls)
     a))
  ([desc operation expected-calls]
   (fn [a]
     (sparse-sv-hint! a desc operation expected-calls)))
  ([operation expected-calls]
   (fn [a desc]
     (sparse-sv-hint! a desc operation expected-calls))))

(defn sparse-sm-hint!
  ([^mkl_rt$sparse_matrix a ^mkl_rt$matrix_descr descr operation
    layout dense-size expected-calls]
   (with-check sparse-error
     (mkl_rt/mkl_sparse_set_sm_hint a (get mkl-sparse-operation operation operation)
                                    descr (get mkl-sparse-layout layout layout)
                                    dense-size expected-calls)
     a))
  ([desc operation layout dense-size expected-calls]
   (fn [a]
     (sparse-sm-hint! a desc operation layout dense-size expected-calls)))
  ([operation layout dense-size expected-calls]
   (fn [a desc]
     (sparse-sm-hint! a desc operation layout dense-size expected-calls))))

(defn optimize!
  ([^mkl_rt$sparse_matrix a]
   (with-check sparse-error
     (mkl_rt/mkl_sparse_optimize a)
     a))
  ([a hints]
   (doseq [hint hints]
     (hint a))
   (optimize! a))
  ([a desc hints]
   (doseq [hint hints]
     (hint a desc))
   (optimize! a)))

(defn sp2m [transa descra a transb descrb b request c]
  (with-check sparse-error
    (mkl_rt/mkl_sparse_sp2m (int transa) ^mkl_rt$matrix_descr descra ^mkl_rt$sparse_matrix a
                            (int transb) ^mkl_rt$matrix_descr descrb ^mkl_rt$sparse_matrix b
                            (int (get mkl-sparse-request request request)) ^mkl_rt$sparse_matrix c)
    c))
