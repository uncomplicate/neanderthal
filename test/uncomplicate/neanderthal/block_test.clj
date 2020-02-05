;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.block-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons.core :refer [release with-release double-fn]]
            [uncomplicate.fluokitten
             [core :refer [fmap! fold foldmap op]]
             [test :refer :all]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [block :refer [buffer contiguous?]]])
  (:import  [clojure.lang IFn$LLD IFn$LD IFn$DD ExceptionInfo]))

(defn test-create [factory]
  (facts "Create tests."
         (with-release [x0 (vctr factory 0)
                        x1 (vctr factory 1)
                        xr0 (vctr factory 0)
                        xr1 (vctr factory 1)
                        a00 (ge factory 0 0)
                        a11 (ge factory 1 1)
                        ar00 (ge factory 0 0)
                        ar11 (ge factory 1 1)]
           (dim x0) => 0
           (dim x1) => 1
           (dim xr0) => 0
           (dim xr1) => 1
           (vctr factory -1) => (throws ExceptionInfo)
           (mrows a00) => 0
           (ncols a00) => 0
           (mrows a11) => 1
           (ncols a11) => 1
           (mrows ar00) => 0
           (ncols ar00) => 0
           (mrows ar11) => 1
           (ncols ar11) => 1
           (ge factory -1 -1) => (throws ExceptionInfo)
           (ge factory -3 0) => (throws ExceptionInfo)
           (tr factory -1) => (throws ExceptionInfo))))

(defn test-equality [factory]
  (facts "Equality and hash code tests."
         (with-release [x1 (vctr factory [1 2 3 4])
                        y1 (vctr factory [1 2 3 4])
                        x2 (row (ge factory 2 2 [1 2 3 4]) 0)
                        y2 (vctr factory [1 3])
                        y3 (vctr factory [1 2 3 5])]
           (.equals x1 nil) => false
           (= x1 y1) => true
           (= x1 y2) => false
           (= x2 y2) => true
           (= x1 y3) => false)))

(defn test-release [factory]
  (let [a (ge factory 2 3 [1 2 3 4 5 6])
        col-a (col a 0)
        sub-a (submatrix a 0 0 1 1)]
    (facts "RealBlockVector and RealBlockMatrix release tests."
           (release col-a) => true
           (release col-a) => true
           (release sub-a) => true
           (release sub-a) => true
           (release a) => true
           (release a) => true)))

(defn test-vctr-transfer [factory0 factory1]
  (with-release [x0 (vctr factory0 4)
                 x1 (vctr factory0 [1 2 3 0])
                 x2 (vctr factory0 [22 33 3 0])
                 y0 (vctr factory1 4)
                 y1 (vctr factory1 [22 33])
                 y2 (vctr factory1 [22 33 3 0])]
    (facts
     "Vector transfer tests."
     (transfer! (float-array [1 2 3]) x0) => x1
     (transfer! (double-array [1 2 3]) x0) => x1
     (transfer! (int-array [1 2 3 0 44]) x0) => x1
     (transfer! (long-array [1 2 3]) x0) => x1
     (seq (transfer! x1 (float-array 2))) => [1.0 2.0]
     (seq (transfer! x1 (double-array 2))) => [1.0 2.0]
     (transfer! y1 x0) => x2
     (transfer! x2 y0) => y2)))

(defn test-ge-transfer [factory0 factory1]
  (with-release [x0 (vctr factory0 6)
                 x1 (vctr factory0 [1 2 3 4 5 6])
                 y0 (vctr factory1 6)
                 y1 (vctr factory1 [1 2 3 4 5 6])
                 a0 (ge factory0 2 3)
                 a1 (ge factory0 2 3 [1 2 3])
                 a2 (ge factory0 2 3 [22 33])
                 b0 (ge factory1 2 3)
                 b1 (ge factory1 2 3 [22 33])
                 b2 (ge factory1 2 3 [22 33])]
    (facts
     "GE transfer tests."
     (transfer! (float-array [1 2 3]) a0) => a1
     (transfer! (double-array [1 2 3]) a0) => a1
     (transfer! (int-array [1 2 3 0]) a0) => a1
     (transfer! (long-array [1 2 3]) a0) => a1
     (seq (transfer! a1 (float-array 2))) => [1.0 2.0]
     (seq (transfer! a1 (double-array 2))) => [1.0 2.0]
     (transfer! b1 a0) => a2
     (transfer! a2 b0) => b2
     (transfer! (transfer! x1 a0) x0) => x1
     (transfer! (transfer! y1 a1) y0) => y1)))

(defn test-tr-transfer [factory0 factory1]
  (with-release [a0 (tr factory0 3)
                 a1 (tr factory0 3 [1 2 3])
                 a2 (tr factory0 3 [22 33])
                 b0 (tr factory1 3)
                 b1 (tr factory1 3 [22 33])
                 b2 (tr factory1 3 [22 33])]
    (facts
     "TR transfer tests."
     (transfer! (float-array [1 2 3]) a0) => a1
     (transfer! (double-array [1 2 3]) a0) => a1
     (transfer! (int-array [1 2 3 0]) a0) => a1
     (transfer! (long-array [1 2 3]) a0) => a1
     (seq (transfer! a1 (float-array 2))) => [1.0 2.0]
     (seq (transfer! a1 (double-array 2))) => [1.0 2.0]
     (transfer! b1 a0) => a2
     (transfer! a2 b0) => b2)))

(defn test-vctr-contiguous [factory]
  (with-release [x (vctr factory 4)
                 a (ge factory 4 4 (range 16))]
    (facts
     "Vector contiguous tests."
     (contiguous? x) => true
     (contiguous? (row a 0)) => false
     (contiguous? (row a 1)) => false
     (contiguous? (col a 0)) => true
     (contiguous? (col a 1)) => true)))

(defn test-ge-contiguous [factory]
  (with-release [a (ge factory 4 4 (range 16))]
    (facts
     "GE contiguous tests."
     (contiguous? a) => true
     (contiguous? (submatrix a 0 0 4 2)) => true
     (contiguous? (submatrix a 0 1 4 2)) => true
     (contiguous? (submatrix a 0 1 3 2)) => false)))

(defn test-uplo-contiguous [factory uplo]
  (with-release [a (uplo factory 4 (range 16))]
    (facts
     "Uplo contiguous tests."
     (contiguous? a) => false)))

;; ================= Vector  ========================================

(defn test-vctr-ifn [factory]
  (facts "IFn implementation for real block vector"
         (with-release [x (vctr factory [1 2 3 4])]
           (x 2) => 3.0
           (x 5) => (throws ExceptionInfo)
           (x -1) => (throws ExceptionInfo)
           (instance? clojure.lang.IFn x) => true
           (.invokePrim ^IFn$LD x 0) => 1.0)))

(defn test-vctr-op [factory]
  (with-release [x (vctr factory [1 2 3])
                 y (vctr factory [4 5])
                 z (vctr factory [])
                 v1 (op x y)
                 v2 (op y x)
                 v3 (op x y x)
                 v4 (op x y x y)
                 v5 (op x y x y z z z x)
                 t1 (vctr factory [1 2 3 4 5])
                 t2 (vctr factory [4 5 1 2 3])
                 t3 (vctr factory [1 2 3 4 5 1 2 3])
                 t4 (vctr factory [1 2 3 4 5 1 2 3 4 5])
                 t5 (vctr factory [1 2 3 4 5 1 2 3 4 5 1 2 3])]
    (facts
     "BlockVector should be a Monoid."
     v1 => t1
     v2 => t2
     v3 => t3
     v4 => t4
     v5 => t5)))

(defn test-vctr-functor [factory]
  (with-release [fx (fn [] (vctr factory [1 2 3 4]))
                 fy (fn [] (vctr factory [2 3 4 5 6]))
                 x (fx)
                 f (fn
                     (^double [^double x] (+ x 1.0))
                     (^double [^double x ^double y] (+ x y))
                     (^double [^double x ^double y ^double z] (+ x y z))
                     (^double [^double x ^double y ^double z ^double w] (+ x y z w)))]

    (facts "Functor implementation for real block vector"
           (instance? IFn$DD f) => true

           (fmap! f (fx)) => (vctr factory [2 3 4 5])
           (fmap! f x) => x

           (fmap! f (fx) (fy)) => (vctr factory [3 5 7 9])
           (fmap! f x (fy)) => x

           (fmap! f (fx) (fy) (fy)) => (vctr factory [5 8 11 14])
           (fmap! f x (fy) (fy)) => x

           (fmap! f (fx) (fy) (fy) (fy)) => (vctr factory [7 11 15 19])
           (fmap! f x (fy) (fy) (fy)) => x

           (fmap! + (fx) (fy) (fy) (fy) [(fy)])
           => (throws ExceptionInfo))))

(defn test-vctr-fold [factory]
  (with-release [x (vctr factory [1 2 3 4])
                 *' (fn ^double [^double x ^double y]
                      (* x y))
                 +' (fn ^double [^double x ^double y]
                      (+ x y))]
    (facts "Fold implementation for vector"

           (fold x) => 10.0
           (fold *' 1.0 x) => 24.0
           (fold +' 0.0 x) => (fold x))))

(defn test-vctr-reducible [factory]
  (with-release [y (vctr factory [2 3 4 5 6])
                 x (vctr factory [1 2 3 4])
                 pf1 (fn ^double [^double res ^double x] (+ x res))
                 pf1o (fn [res ^double x] (conj res x))]
    (facts "Reducible implementation for vector"

           (fold pf1 1.0 x) => 11.0
           (fold pf1o [] x) => [1.0 2.0 3.0 4.0]

           (fold pf1 1.0 x y) => 25.0
           (fold pf1o [] x y) => [3.0 5.0 7.0 9.0]

           (fold pf1 1.0 x y y) => 39.0
           (fold pf1o [] x y y) => [5.0 8.0 11.0 14.0]

           (fold pf1 1.0 x y y y) => 53.0
           (fold pf1o [] x y y y) => [7.0 11.0 15.0 19.0]

           (fold + 1.0 x y y y) => 53.0)))

(defn test-vctr-seq [factory]
  (facts "Vector as a sequence"
         (seq (vctr factory [1 2 3])) => '(1.0 2.0 3.0)
         (seq (row (ge factory 2 3 (range 6)) 1)) => '(1.0 3.0 5.0)))

;; ================= GE matrix ========================================

(defn test-ge-ifn [factory]
  (facts "IFn implementation for double general matrix"
         (with-release [x (ge factory 2 3 [1 2 3 4 5 6])]
           (x 1 2) => 6.0
           (x 2 1) => (throws ExceptionInfo)
           (x -1 3) => (throws ExceptionInfo)
           (instance? clojure.lang.IFn x) => true
           (.invokePrim ^IFn$LLD x 0 0) => 1.0)))

(defn test-ge-op [factory]
  (facts
   "GeneralMatrix should be a Monoid."
   (with-release [x (ge factory 2 3 (range 6))
                  y (ge factory 2 2 [6 7 8 9])
                  z (ge factory 0 0 [])
                  v1 (op x y)
                  v2 (op y x)
                  v3 (op x y x)
                  v4 (op x y x y)
                  v5 (op x y x y z z z x)
                  t1 (ge factory 2 5 (range 10))
                  t2 (ge factory 2 5 [6 7 8 9 0 1 2 3 4 5])
                  t3 (ge factory 2 8 (op (range 6) [6 7 8 9] (range 6)))
                  t4 (ge factory 2 10 (op (range 10) (range 10)))
                  t5 (ge factory 2 13 (op (range 10) (range 10) (range 6)))]
     v1 => t1
     v2 => t2
     v3 => t3
     v4 => t4
     v5 => t5)))

(defn test-ge-functor [factory]
  (with-release [fx (fn [] (ge factory 2 3 [1 2 3 4 5 6]))
                 fy (fn [] (ge factory 2 3 [2 3 4 5 6 7]))
                 x (fx)
                 f (fn
                     (^double [^double x] (+ x 1.0))
                     (^double [^double x ^double y] (+ x y))
                     (^double [^double x ^double y ^double z] (+ x y z))
                     (^double [^double x ^double y ^double z ^double w] (+ x y z w)))]
    (facts "Functor implementation for real GE matrix"
           (instance? clojure.lang.IFn$DD f) => true

           (fmap! f (fx)) => (fy)
           (fmap! f x) => x

           (fmap! f (fx) (fy)) => (ge factory 2 3 [3 5 7 9 11 13])
           (fmap! f x (fy)) => x

           (fmap! f (fx) (fy) (fy)) => (ge factory 2 3 [5 8 11 14 17 20])
           (fmap! f x (fy) (fy)) => x

           (fmap! f (fx) (fy) (fy) (fy)) => (ge factory 2 3 [7 11 15 19 23 27])
           (fmap! f x (fy) (fy) (fy)) => x

           (fmap! + (fx) (fy) (fy) (fy) [(fy)]) => (throws ExceptionInfo))))

(defn test-ge-fold [factory]
  (with-release [x (ge factory 2 3 [1 2 3 4 5 6])
                 *' (fn ^double [^double x ^double y] (double (* x y)))
                 +' (fn ^double [^double x ^double y] (double (+ x y)))]
    (facts "Fold implementation for real GE matrix"
           (fold x) => 21.0
           (fold *' 1.0 x) => 720.0
           (fold +' 0.0 x) => (fold x))))

(defn test-ge-reducible [factory]
  (with-release [x (ge factory 2 3 [1 2 3 4 5 6])
                 y (ge factory 2 3 [2 3 4 5 6 7])
                 pf1 (fn ^double [^double res ^double x] (+ x res))
                 pf1o (fn [res ^double x] (conj res x))]
    (facts "Reducible implementation for real GE matrix"

           (fold pf1 1.0 x) => 22.0
           (fold pf1o [] x) => [1.0 2.0 3.0 4.0 5.0 6.0]

           (fold pf1 1.0 x y) => 49.0
           (fold pf1o [] x y) => [3.0 5.0 7.0 9.0 11.0 13.0]

           (fold pf1 1.0 x y y) => 76.0
           (fold pf1o [] x y y) => [5.0 8.0 11.0 14.0 17.0 20.0]

           (fold + 1.0 x y y y) => 103.0)))


(defn test-ge-seq [factory]
  (facts "GE matrix as a sequence"

         (seq (ge factory 2 3 (range 6) {:layout :column}))
         => (list (list 0.0 1.0) (list 2.0 3.0) (list 4.0 5.0))

         (seq (ge factory 2 3 (range 6) {:layout :row}))
         => (list (list 0.0 1.0 2.0) (list 3.0 4.0 5.0))))

;; ================= TR matrix ========================================

(defn test-tr-functor [factory]
  (with-release [fx (fn [] (tr factory 3 [1 2 3 4 5 6]))
                 fy (fn [] (tr factory 3 [2 3 4 5 6 7]))
                 x (fx)
                 f (fn
                     (^double [^double x] (+ x 1.0))
                     (^double [^double x ^double y] (+ x y))
                     (^double [^double x ^double y ^double z] (+ x y z))
                     (^double [^double x ^double y ^double z ^double w] (+ x y z w)))]
    (facts "Functor implementation for real TR matrix"
           (instance? clojure.lang.IFn$DD f) => true

           (fmap! f (fx)) => (fy)
           (fmap! f x) => x

           (fmap! f (fx) (fy)) => (tr factory 3 [3 5 7 9 11 13])
           (fmap! f x (fy)) => x

           (fmap! f (fx) (fy) (fy)) => (tr factory 3 [5 8 11 14 17 20])
           (fmap! f x (fy) (fy)) => x

           (fmap! f (fx) (fy) (fy) (fy)) => (tr factory 3 [7 11 15 19 23 27])
           (fmap! f x (fy) (fy) (fy)) => x

           (fmap! + (fx) (fy) (fy) (fy) [(fy)]) => (throws ExceptionInfo))))

(defn test-tr-fold [factory]
  (with-release [x (tr factory 3 [1 2 3 4 5 6])
                 *' (fn ^double [^double x ^double y] (double (* x y)))
                 +' (fn ^double [^double x ^double y] (double (+ x y)))]
    (facts "Fold implementation for real TR matrix"
           (fold x) => 21.0
           (fold *' 1.0 x) => 720.0
           (fold +' 0.0 x) => (fold x))))

(defn test-tr-reducible [factory]
  (with-release [x (tr factory 3 [1 2 3 4 5 6])
                 y (tr factory 3 [2 3 4 5 6 7])
                 pf1 (fn ^double [^double res ^double x] (+ x res))
                 pf1o (fn [res ^double x] (conj res x))]
    (facts "Reducible implementation for real TR matrix"

           (fold pf1 1.0 x) => 22.0
           (fold pf1o [] x) => [1.0 2.0 3.0 4.0 5.0 6.0]

           (fold pf1 1.0 x y) => 49.0
           (fold pf1o [] x y) => [3.0 5.0 7.0 9.0 11.0 13.0]

           (fold pf1 1.0 x y y) => 76.0
           (fold pf1o [] x y y) => [5.0 8.0 11.0 14.0 17.0 20.0]

           (fold + 1.0 x y y y) => 103.0)))

(defn test-tr-seq [factory]
  (facts "TR matrix as a sequence"

         (seq (tr factory 3 (range 1 7) {:layout :column}))
         => (list (list 1.0 2.0 3.0) (list 4.0 5.0) (list 6.0))

         (seq (tr factory 3 (range 1 7) {:layout :column :uplo :upper}))
         => (list (list 1.0) (list 2.0 3.0) (list 4.0 5.0 6.0))

         (seq (tr factory 3 (range 1 7) {:layout :column :uplo :upper :diag :unit}))
         => (list (list) (list 1.0) (list 2.0 3.0))

         (seq (tr factory 3 (range 1 7) {:layout :row}))
         => (list (list 1.0) (list 2.0 3.0) (list 4.0 5.0 6.0))

         (seq (tr factory 3 (range 1 7) {:layout :row :uplo :upper}))
         => (list (list 1.0 2.0 3.0) (list 4.0 5.0) (list 6.0))

         (seq (tr factory 3 (range 1 7) {:layout :row :uplo :upper :diag :unit}))
         => (list (list 1.0 2.0) (list 3.0) (list))

         (seq (tr factory 3 (range 1 7) {:layout :row :uplo :lower :diag :unit}))
         => (list (list) (list 1.0) (list 2.0 3.0))))

(defn test-vctr-functor-laws [factory]
  (with-release [pinc (double-fn (partial + 1))
                 p*100 (double-fn (partial * 100))
                 p+ (double-fn +)]
    (facts "Monadic laws for vector."
           (functor-law2 pinc p*100 (vctr factory [1 -199 9]))
           (functor-law2 pinc p*100 (vctr factory [1 -199 9] (vctr factory [1 -66 9])))
           (fmap-keeps-type pinc (vctr factory [100 0 -999]))
           (fmap-keeps-type p+ (vctr factory [100 0 -999]) (vctr factory (list 44 0 -54))))))

(defn test-ge-functor-laws [factory creator]
  (with-release [pinc (double-fn (partial + 1))
                 p*100 (double-fn (partial * 100))
                 p+ (double-fn +)]
    (facts "Monadic laws for GE matrices."
           (functor-law2 pinc p*100 (creator factory 3 2 [1 -199 9 7 8 19]))
           (functor-law2 pinc p*100 (creator factory 2 3 [1 -199 9 8 -7 -1])
                         (creator factory 2 3 [9 8 -7 1 -66 9]))
           (fmap-keeps-type pinc (creator factory 1 9 [1 2 3 4 5 6 100 0 -999]))
           (fmap-keeps-type p+ (creator factory 0 0) (ge factory 0 0)))))

(defn test-sspar-mat-functor-laws [factory creator]
  (with-release [pinc (double-fn (partial + 1))
                 p*100 (double-fn (partial * 100))
                 p+ (double-fn +)]
    (facts "Monadic laws for UPLO and packed matrices."
           (functor-law2 pinc p*100 (creator factory 3 [1 -199 9 7 8 19]))
           (functor-law2 pinc p*100 (creator factory 2 [1 -199 9] (creator factory 2 [9 8 -7])))
           (fmap-keeps-type pinc (creator factory 5 [1 2 3 4 5 6 100 0 -999]))
           (fmap-keeps-type p+ (creator factory 0) (creator factory 0)))))

(defn test-extend-buffer [factory]
  (with-release [x (vctr factory (range 16))
                 y (vctr factory (range 16))]
    (view (buffer x)) => y))

(defn test-all [factory]
  (test-create factory)
  (test-equality factory)
  (test-release factory)
  (test-vctr-op factory)
  (test-ge-op factory)
  (test-vctr-contiguous factory)
  (test-ge-contiguous factory)
  (test-uplo-contiguous factory tr))

(defn test-both-factories [factory0 factory1]
  (test-vctr-transfer factory0 factory1)
  (test-ge-transfer factory0 factory1)
  (test-tr-transfer factory0 factory1))

(defn test-host [factory]
  (test-uplo-contiguous factory sy)
  (test-vctr-ifn factory)
  (test-vctr-functor factory)
  (test-vctr-fold factory)
  (test-vctr-reducible factory)
  (test-vctr-seq factory)
  (test-ge-ifn factory)
  (test-ge-functor factory)
  (test-ge-fold factory)
  (test-ge-reducible factory)
  (test-ge-seq factory)
  (test-tr-functor factory)
  (test-tr-fold factory)
  (test-tr-reducible factory)
  (test-tr-seq factory)
  (test-vctr-functor-laws factory)
  (test-ge-functor-laws factory ge)
  (test-ge-functor-laws factory gb)
  (test-sspar-mat-functor-laws factory tr)
  (test-sspar-mat-functor-laws factory sy)
  (test-sspar-mat-functor-laws factory sb)
  (test-sspar-mat-functor-laws factory sp)
  (test-sspar-mat-functor-laws factory tp)
  (test-sspar-mat-functor-laws factory gd)
  (test-sspar-mat-functor-laws factory gt)
  (test-sspar-mat-functor-laws factory st))
