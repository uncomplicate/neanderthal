;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.sparse-test
  (:require [midje.sweet :refer [facts throws =>]]
            [uncomplicate.commons.core :refer [with-release release view]]
            [uncomplicate.fluokitten.core :refer [fmap fmap! fold foldmap]]
            [uncomplicate.neanderthal.core :refer :all]
            [uncomplicate.neanderthal.sparse :refer [csv csr csv? csr?]]
            [uncomplicate.neanderthal.internal.cpp.structures
             :refer [entries indices columns indexb indexe]] ;;TODO
            [uncomplicate.neanderthal.internal.cpp.mkl
             [factory :refer [mkl-float mkl-double mkl-int mkl-long mkl-short mkl-byte]]])
  (:import  clojure.lang.ExceptionInfo))

;; ================= Sparse tests ===============================

(defn test-create [factory]
  (facts "Sparse constructors."
         (with-release [x0 (csv factory 0)
                        x1 (csv factory 1)
                        xr0 (csv factory 0)
                        xr1 (csv factory 1)
                        a00 (csr factory 0 0)
                        a11 (csr factory 1 1)]
           (dim x0) => 0
           (seq (entries x0)) => []
           (seq (indices x0)) => []
           (dim x1) => 1
           (seq (entries x1)) => []
           (seq (indices x1)) => []
           (dim xr0) => 0
           (dim xr1) => 1
           (csv factory -1) => (throws ExceptionInfo)
           (mrows a00) => 0
           (ncols a00) => 0
           (mrows a11) => 1
           (ncols a11) => 1
           (csv factory -1) => (throws ExceptionInfo)
           (csr factory -1 -1) => (throws ExceptionInfo)
           (csr factory -3 0) => (throws ExceptionInfo))))

(defn test-equality [factory]
  (facts "Sparse equality and hash code tests."
         (with-release [x1 (csv factory 70 [10 20 30 40])
                        y1 (csv factory 70 [10 20 30 40])
                        y2 (csv factory 70 [1 3])
                        y3 (csv factory 60 [10 20 30 40])
                        y4 (csv factory 70 [10 20 30 40] [1.0])
                        y5 (csv factory 70)
                        a1 (csr factory 1 70 [[10 20 30 40] nil])
                        a2 (csr factory 1 70 [[10 20 30 40] nil])
                        a3 (csr factory 1 70 [[1 3] nil])
                        a4 (csr factory 1 60 [[10 20 30 40] nil])
                        a5 (csr factory 2 70)]
           (.equals x1 nil) => false
           (= x1 x1) => true
           (= x1 y1) => true
           (= x1 y2) => false
           (= x1 y3) => false
           (= x1 y4) => false
           (= x1 y5) => false
           (= a1 a1) => true
           (= a1 a2) => true
           (= a1 a3) => false
           (= a1 a4) => false
           (csr factory 2 70 [[1 3] nil]) => (throws ExceptionInfo))))

(defn test-release [factory]
  (let [x (csv factory 3 [1] [1.0])
        a (csr factory 2 3 [[1] [1.0]
                            [1] [2.0]])
        b (csr factory 3 2 [[1] [1.0]
                            [1] [2.0]]
               {:layout :column})
        row-a (row a 1)
        col-b (col b 0)
        sub-a (submatrix a 1 0 1 3)
        sub-b (submatrix b 1 0 1 2)]
    (facts "CSVector and CSRMatrix release tests."
           (release row-a) => true
           (release row-a) => true
           (release col-b) => true
           (release col-b) => true
           (release sub-a) => true
           (release sub-a) => true
           (release sub-b) => true
           (release sub-b) => true
           (release x) => true
           (release x) => true
           (release a) => true
           (release a) => true)))

(defn test-csv-transfer [factory0 factory1]
  (with-release [x0 (csv factory0 4 [1 3])
                 x1 (csv factory0 4 [1 3] [1.0 2.0])
                 x2 (csv factory0 4 [1 3] [10 20])
                 y0 (csv factory1 55 [22 33])
                 y1 (csv factory1 55 [22 33] [10 20])
                 y2 (csv factory1 55 [22 33] [10 20])]
    (facts
     "Vector transfer tests."
     (transfer! (float-array [1 2 3]) x0) => x1
     (transfer! (double-array [1 2 3]) x0) => x1
     (transfer! (int-array [1 2 3 0 44]) x0) => x1
     (transfer! (long-array [1 2 3]) x0) => x1
     (seq (transfer! x1 (float-array 2))) => [1.0 2.0]
     (seq (transfer! x1 (double-array 2))) => [1.0 2.0]
     (transfer! x1 (int-array 2)) => (throws ExceptionInfo)
     (transfer! x1 (long-array 2)) => (throws ExceptionInfo)
     (transfer! x1 (short-array 2)) => (throws ExceptionInfo)
     (transfer! x1 (byte-array 2)) => (throws ExceptionInfo)
     (transfer! y1 x0) => x2
     (transfer! x2 y0) => y2)))

(defn test-csr-transfer [factory0 factory1]
  (with-release [x0 (csv factory0 4 [1 3])
                 x1 (csv factory0 4 [1 3] [1.0 2.0])
                 y0 (csv factory1 55 [22 33])
                 y1 (csv factory1 55 [22 33] [10 20])
                 a0 (csr factory0 2 4 [[1 3] []
                                       [0 2] []])
                 a1 (csr factory0 2 4 [[1 3] [1.0 2.0]
                                       [0 2] []])
                 a2 (csr factory0 2 4 [[1 3] [10 20]
                                       [0 2] [100 200]])
                 b0 (csr factory1 2 4 [[1 3] []
                                       [0 2] []])
                 b1 (csr factory1 2 4 [[1 3] [10 20]
                                       [0 2] [100 200]])
                 b2 (csr factory1 2 4 [[1 3] [10 20]
                                       [0 2] [100 200]])]
    (facts
     "CSR transfer tests."
     (transfer! (float-array [1 2]) a0) => a1
     (transfer! (double-array [1 2]) a0) => a1
     (transfer! (int-array [1 2 0 0 44]) a0) => a1
     (transfer! (long-array [1 2]) a0) => a1
     (seq (transfer! a1 (float-array 2))) => [1.0 2.0]
     (seq (transfer! a1 (double-array 2))) => [1.0 2.0]
     (transfer! a1 (int-array 2)) => (throws ExceptionInfo)
     (transfer! a1 (long-array 2)) => (throws ExceptionInfo)
     (transfer! a1 (short-array 2)) => (throws ExceptionInfo)
     (transfer! a1 (byte-array 2)) => (throws ExceptionInfo)
     (transfer! b1 a0) => a2
     (transfer! a2 b0) => b2
     (transfer! (transfer! x1 a0) x0) => x1
     (transfer! (transfer! y1 a1) y0) => y1)))


;; =================== Fluokitten ============================

(defn test-csv-functor [factory]
  (with-release [fx (fn [] (csv factory 10 [1 3 5 7] [1 2 3 4]))
                 fy (fn [] (csv factory 10 [1 3 5 7] [2 3 4 5]))
                 fz (fn [] (csv factory 10 [1 3 5 7 9] [2 3 4 5 6]))
                 x (fx)
                 f (fn
                     (^double [^double x] (+ x 1.0))
                     (^double [^double x ^double y] (+ x y))
                     (^double [^double x ^double y ^double z] (+ x y z))
                     (^double [^double x ^double y ^double z ^double w] (+ x y z w)))]

    (facts "Functor implementation for compressed sparse vector"

           (fmap! f (fx)) => (csv factory 10 [1 3 5 7] [2 3 4 5])
           (fmap! f x) => x

           (fmap! f (fx) (fy)) => (csv factory 10 [1 3 5 7] [3 5 7 9])
           (fmap! f x (fy)) => x

           (fmap! f (fx) (fy) (fy)) => (csv factory 10 [1 3 5 7] [5 8 11 14])
           (fmap! f x (fy) (fy)) => x

           (fmap! f (fx) (fy) (fy) (fy)) => (csv factory 10 [1 3 5 7] [7 11 15 19])
           (fmap! f x (fy) (fy) (fy)) => x

           (fmap! + (fx) (fy) (fy) (fy) [(fy)]) => (throws Exception)
           (fmap! + (fx) (fz)) => (throws ExceptionInfo))))

(defn test-csv-fold [factory]
  (with-release [x (csv factory 10 [1 3 5 7] [1 2 3 4])
                 *' (fn ^double [^double x ^double y]
                      (* x y))
                 +' (fn ^double [^double x ^double y]
                      (+ x y))]
    (facts "Fold implementation for compressed sparse vector."

           (fold x) => 10.0
           (fold *' 1.0 x) => 24.0
           (fold +' 0.0 x) => (fold x))))

(defn test-csv-reducible [factory]
  (with-release [y (csv factory 10 [1 3 5 7 9] [2 3 4 5 6])
                 x (csv factory 10 [1 3 5 7 9] [1 2 3 4 0])
                 pf1 (fn ^double [^double res ^double x] (+ x res))
                 pf1o (fn [res ^double x] (conj res x))]
    (facts "Reducible implementation for vector"

           (fold pf1 1.0 x) => 11.0
           (fold pf1o [] x) => [1.0 2.0 3.0 4.0 0.0]

           (fold pf1 1.0 x y) => 31.0
           (fold pf1o [] x y) => [3.0 5.0 7.0 9.0 6.0]

           (fold pf1 1.0 x y y) => 51.0
           (fold pf1o [] x y y) => [5.0 8.0 11.0 14.0 12.0]

           (fold pf1 1.0 x y y y) => 71.0
           (fold pf1o [] x y y y) => [7.0 11.0 15.0 19.0 18.0]

           (fold + 1.0 x y y y) => 71.0)))

(defn test-csv-seq [factory]
  (facts "Compressed sparse vector as a sequence."
         (seq (csv factory 10 [1 3 5] [1 2 3])) => '(1.0 2.0 3.0)
         (seq (row (csr factory 2 4 [[1 3] [10 20] [0 2] [100 200]]) 1)) => [100.0 200.0]))

;; =================== Neanderthal core  ============================

(defn test-csv-constructor [factory]
  (facts "Create a compressed sparse vector."
         (csv? (csv factory 10 [1 3 5] [1 2 3])) => true
         (csv factory 10 [1 3 5] 1 2 3) => (csv factory 10 [1 3 5] [1 2 3])
         (csv factory 3) => (csv factory 3 [])
         (view (csv factory 10 [1 3 5] 1 2 3)) => (csv factory 10 [1 3 5] 1 2 3)
         (view-vctr (csv factory 10 [1 3 5] (range 8))) => (vctr factory [0 1 2])
         (view-vctr (csv factory 10 [1 3 5] (range 8)) 3) => (vctr factory [0])
         (view-ge (csv factory 10 [1 3 5] 1 2 3 4)) => (ge factory 3 1 [1 2 3])
         (csv factory 10 nil) => (csv factory 10)
         (dim (csv factory 0 [])) => 0
         (csv factory 10 [1 3 5]) => (zero (csv factory 10 [1 3 5] [1 2 3]))))

(defn test-csr-constructor [factory] ;; TODO
  (facts "Create a compressed sparse matrix."

         (csr factory 2 3 [[30] [0] [1] [0]]) => (throws ExceptionInfo)))

(defn test-csr-mv [factory]
  (facts "BLAS 2 GE CSR mv!"
         (mv! 2.0 (csr factory 3 2 [[1] [1.0] [0] [1.0] [2] [0.0]] )
              (vctr factory 1 2 3) 3 (vctr factory [1 2 3 4])) => (throws ExceptionInfo)

         (with-release [y (vctr factory [1 2 3])]
           (identical? (mv! 2 (csr factory 3 2 [[0] [1] [0] [2] [0] [3]]) (vctr factory 1 2) 3 y) y))
         => true

         (mv! (csr factory 2 3 [[0 1 2] [1 2 3] [0 1 2] [10 20 30]])
              (vctr factory 7 0 4) (vctr factory 0 0)) => (vctr factory 19 190)

         (mv! 2.0 (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]])
              (vctr factory 1 2 3) 3.0 (vctr factory 1 2)) => (vctr factory 47 62)

         (mv! 2.0 (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]])
              (vctr factory 1 2 3) 0.0 (vctr factory 0 0))
         => (mv 2.0 (ge factory 2 3 [1 2 3 4 5 6]) (vctr factory 1 2 3))

         ;; (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
         ;;      (vctr factory [1 3 5]) 3.0 (vctr factory [2 3]))
         ;; => (vctr factory 272 293)

         ;; (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
         ;;      (vctr factory [1 3 5]) 3.0 (col (ge factory 2 3 (range 6)) 1))
         ;; => (vctr factory 272 293)

         ;; (mv! 2.0 (submatrix (ge factory 4 6 (range 24)) 1 2 2 3)
         ;;      (row (ge factory 2 3 (range 6)) 1) 3.0 (col (ge factory 2 3 (range 6)) 1))
         ;; => (vctr factory 272 293)

         (mv! (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]] {:layout :row})
              (vctr factory 1 2 3) (vctr factory 0 0))
         => (mv! (csr factory 2 3 [[0 1] [1 2] [0 1] [3 4] [0 1] [5 6]] {:layout :column})
                 (vctr factory 1 2 3) (vctr factory 0 0))

         (mv 2.0 (csr factory 2 30 [[1 2 4] [1 3 5] [1 2 4] [2 4 6]] {:layout :row})
             (vctr factory (into [0 1 2 0 3] (repeat 25 0))) (vctr factory 0 0))
         => (vctr factory 44 56)))

(defn test-csr-mm [factory]
  (facts "BLAS 3 GE CSR mm!"
         (mm! 2.0 (csr factory 3 2 [[0 1] [1 2] [0 1] [3 4] [0 1] [5 6]]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws ExceptionInfo)

         ;; (mm! 2.0 (ge factory 3 2 [1 2 3 4 5 6]) (ge factory 2 3 [1 2 3 4 5 6])
         ;;      3.0 (ge factory 3 2 [1 2 3 4 5 6])) => (throws ExceptionInfo)

         ;; (with-release [c (ge factory 2 2 [1 2 3 4])]
         ;;   (identical? (mm! 2.0 (ge factory 2 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6]) 3.0 c) c))
         ;; => true

         (mm! 2.0 (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]]) (ge factory 3 2 [1 2 3 4 5 6])
              3.0 (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [47 62 107 140])

         (mm! 2.0 (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]])
              (csr factory 3 2 [[0 1] [1 4] [0 1] [2 5] [0 1] [3 6]])
              0.0 (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [44 56 98 128])

         (mm! 2.0 (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]])
              (csr factory 3 2 [[0 1] [1 4] [0 1] [2 5] [0 1] [3 6]])
              3.0 (ge factory 2 2 [1 2 3 4])) => (throws ExceptionInfo)

         (mm! 2.0 (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]])
              (csr factory 3 2 [[0 1] [1 4] [0 1] [2 5] [0 1] [3 6]]))
         => (csr factory 2 2 [[0 1] [44 98] [0 1] [56 128]])

          (mm! 2.0 (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]])
               (csr factory 3 2 [[0 1] [1 4] [0 1] [2 5] [0 1] [3 6]]))
          => (csr factory 2 2 [[0 1] [44 98] [0 1] [56 128]])

          (let [a (csr factory 2 3 [[0 1 2] [1 3 5] [0 1 2] [2 4 6]])
                b (csr factory 3 2 [[0 1] [1 4] [0 1] [2 5] [0 1] [3 6]])
                res (csr a b {:index true})]
            (seq (columns res)) => [0 1 0 1]
            (seq (indexb res)) => [0 2]
            (seq (indexe res)) => [2 4]
            (mm! 2.0 a b 0.0 res) => res
            (seq (entries res)) => [44.0 98.0 56.0 128.0])

         ;; (mm! 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:layout :row}) (ge factory 3 2 [1 4 2 5 3 6] {:layout :row})
         ;;      3.0  (ge factory 2 2 [1 2 3 4])) => (ge factory 2 2 [47 62 107 140])

         ;; (mm 2.0 (ge factory 2 3 [1 3 5 2 4 6] {:layout :row}) (ge factory 3 2 [1 4 2 5 3 6] {:layout :row})
         ;;     3.0  (ge factory 2 2 [1 2 3 4])) => (throws ClassCastException)

         ;; (mm (ge factory 2 3 [1 2 3 4 5 6]) (ge factory 3 2 [1 2 3 4 5 6]))
         ;; => (ge factory 2 2 (list 22.0 28.0 49.0 64.0))

         ;; (with-release [a (ge factory 2 3 (range 1 7 0.1))
         ;;                b (ge factory 3 2 (range 0.1 9 0.2))
         ;;                c (ge factory 2 3 (range 3 5 0.11))
         ;;                d (ge factory 3 2 (range 2 19))
         ;;                ab (mm a b)
         ;;                cd (mm c d)
         ;;                abcd (mm ab cd)
         ;;                abcd-comp (mm a b c d)]
         ;;   abcd-comp => abcd)

         ;; (with-release [a (ge factory 2 2 (range 1 7 0.1))
         ;;                b (ge factory 2 2 (range 0.1 9 0.2))
         ;;                c (ge factory 2 2 (range 3 5 0.11))
         ;;                d (ge factory 2 2 (range 2 19))
         ;;                ab (mm a b)
         ;;                abc (mm ab c)
         ;;                abcd (mm abc d)
         ;;                abcd-comp (mm a b c d)]
         ;;   abcd-comp => abcd
         ))

(defn test-block [factory0 factory1]
  (test-create factory0)
  (test-equality factory0)
  (test-release factory0)
  (test-csv-transfer factory0 factory0)
  (test-csv-transfer factory0 factory1)
  (test-csr-transfer factory0 factory0)
  (test-csr-transfer factory0 factory1)
  (test-csv-functor factory0)
  (test-csv-fold factory0)
  (test-csv-reducible factory0)
  (test-csv-seq factory0))

(defn test-core [factory]
  (test-csv-constructor factory)
  (test-csr-constructor factory)
  (test-csr-mv factory)
  (test-csr-mm factory))

(defn test-all [factory0 factory1]
  (test-block factory0 factory1)
  (test-core factory0))

(test-all mkl-float mkl-double)
(test-all mkl-double mkl-float)
