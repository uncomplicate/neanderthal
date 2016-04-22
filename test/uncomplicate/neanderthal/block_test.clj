(ns uncomplicate.neanderthal.block-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons.core :refer [release with-release]]
            [uncomplicate.fluokitten.core :refer [fmap! fold foldmap op]]
            [uncomplicate.neanderthal.core :refer :all])
  (:import  [clojure.lang IFn$LLD IFn$LD IFn$DD]))

(defn test-equality [factory]
  (facts "Equality and hash code."
         (with-release [x1 (create-vector factory [1 2 3 4])
                        y1 (create-vector factory [1 2 3 4])
                        x2 (row (create-ge-matrix factory 2 2 [1 2 3 4]) 0)
                        y2 (create-vector factory [1 3])
                        y3 (create-vector factory [1 2 3 5])]
           (.equals x1 nil) => false
           (= x1 y1) => true
           (= x1 y2) => false
           (= x2 y2) => true
           (= x1 y3) => false)))

(defn test-release [factory]
  (let [a (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
        col-a (col a 0)
        sub-a (submatrix a 0 0 1 1)]
    (facts "RealBlockVector and RealBlockMatrix release."
           (release col-a) => true
           (release col-a) => true
           (release sub-a) => true
           (release sub-a) => true
           (release a) => true
           (release a) => true)))

(defn test-ifn-vector [factory]
  (facts "IFn implementation for real block vector"
         (let [x (create-vector factory [1 2 3 4])]
           (x 2) => 3.0
           (x 5) => (throws IndexOutOfBoundsException)
           (x -1) => (throws IndexOutOfBoundsException)
           (instance? clojure.lang.IFn x) => true
           (.invokePrim ^IFn$LD x 0) => 1.0)))

(defn test-op-vector [factory]
  (facts
   "BlockVector should be a Monoid."
   (with-release [x (create-vector factory [1 2 3])
                  y (create-vector factory [4 5])
                  z (create-vector factory [])
                  v1 (op x y)
                  v2 (op y x)
                  v3 (op x y x)
                  v4 (op x y x y)
                  v5 (op x y x y z z z x)
                  t1 (create-vector factory [1 2 3 4 5])
                  t2 (create-vector factory [4 5 1 2 3])
                  t3 (create-vector factory [1 2 3 4 5 1 2 3])
                  t4 (create-vector factory [1 2 3 4 5 1 2 3 4 5])
                  t5 (create-vector factory [1 2 3 4 5 1 2 3 4 5 1 2 3])]
     v1 => t1
     v2 => t2
     v3 => t3
     v4 => t4
     v5 => t5)))

(defn test-functor-vector [factory]
  (let [fx (fn [] (create-vector factory [1 2 3 4]))
        fy (fn [] (create-vector factory [2 3 4 5 6]))
        x (fx)
        f (fn
            (^double [^double x] (+ x 1.0))
            (^double [^double x ^double y] (+ x y))
            (^double [^double x ^double y ^double z] (+ x y z))
            (^double [^double x ^double y ^double z ^double w] (+ x y z w)))]

    (facts "Functor implementation for real block vector"
           (instance? IFn$DD f) => true

           (fmap! f (fx)) => (create-vector factory [2 3 4 5])
           (fmap! f x) => x

           (fmap! f (fx) (fy)) => (create-vector factory [3 5 7 9])
           (fmap! f x (fy)) => x

           (fmap! f (fx) (fy) (fy)) => (create-vector factory [5 8 11 14])
           (fmap! f x (fy) (fy)) => x

           (fmap! f (fx) (fy) (fy) (fy)) => (create-vector factory [7 11 15 19])
           (fmap! f x (fy) (fy) (fy)) => x

           (fmap! + (fx) (fy) (fy) (fy) [(fy)])
           => (throws UnsupportedOperationException))))

(defn test-fold-vector [factory]
  (let [x (create-vector factory [1 2 3 4])
        *' (fn ^double [^double x ^double y]
             (* x y))
        +' (fn ^double [^double x ^double y]
             (+ x y))]
    (facts "Fold implementation for vector"

           (fold x) => 10.0
           (fold *' 1.0 x) => 24.0
           (fold +' 0.0 x) => (fold x))))

(defn test-reducible-vector [factory]
  (let [y (create-vector factory [2 3 4 5 6])
        x (create-vector factory [1 2 3 4])
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

           (fold + 1.0 x y y y) => (throws IllegalArgumentException))))

(defn test-seq-vector [factory]
  (facts "Vector as a sequence"
         (seq (create-vector factory [1 2 3])) => '(1.0 2.0 3.0)
         (seq (row (create-ge-matrix factory 2 3 (range 6)) 1)) => '(1.0 3.0 5.0)))

(defn test-ifn-ge-matrix [factory]
  (facts "IFn implementation for double general matrix"
         (let [x (create-ge-matrix factory 2 3 [1 2 3 4 5 6])]
           (x 1 2) => 6.0
           (x 2 1) => (throws IndexOutOfBoundsException)
           (x -1 3) => (throws IndexOutOfBoundsException)
           (instance? clojure.lang.IFn x) => true
           (.invokePrim ^IFn$LLD x 0 0) => 1.0)))

(defn test-op-ge-matrix [factory]
  (facts
   "GeneralMatrix should be a Monoid."
   (with-release [x (create-ge-matrix factory 2 3 (range 6))
                  y (create-ge-matrix factory 2 2 [6 7 8 9])
                  z (create-ge-matrix factory 2 0 [])
                  v1 (op x y)
                  v2 (op y x)
                  v3 (op x y x)
                  v4 (op x y x y)
                  v5 (op x y x y z z z x)
                  t1 (create-ge-matrix factory 2 5 (range 10))
                  t2 (create-ge-matrix factory 2 5 [6 7 8 9 0 1 2 3 4 5])
                  t3 (create-ge-matrix factory 2 8 (op (range 6) [6 7 8 9] (range 6)))
                  t4 (create-ge-matrix factory 2 10 (op (range 10) (range 10)))
                  t5 (create-ge-matrix factory 2 13 (op (range 10) (range 10) (range 6)))]
     v1 => t1
     v2 => t2
     v3 => t3
     v4 => t4
     v5 => t5)))

(defn test-functor-ge-matrix [factory]
  (let [fx (fn [] (create-ge-matrix factory 2 3 [1 2 3 4 5 6]))
        fy (fn [] (create-ge-matrix factory 2 3 [2 3 4 5 6 7]))
        x (fx)
        f (fn
            (^double [^double x] (+ x 1.0))
            (^double [^double x ^double y] (+ x y))
            (^double [^double x ^double y ^double z] (+ x y z))
            (^double [^double x ^double y ^double z ^double w] (+ x y z w)))]
    (facts "Functor implementation for real general matrix"
           (instance? clojure.lang.IFn$DD f) => true

           (fmap! f (fx)) => (fy)
           (fmap! f x) => x

           (fmap! f (fx) (fy))
           => (create-ge-matrix factory 2 3 [3 5 7 9 11 13])
           (fmap! f x (fy)) => x

           (fmap! f (fx) (fy) (fy))
           => (create-ge-matrix factory 2 3 [5 8 11 14 17 20])
           (fmap! f x (fy) (fy)) => x

           (fmap! f (fx) (fy) (fy) (fy))
           => (create-ge-matrix factory 2 3 [7 11 15 19 23 27])
           (fmap! f x (fy) (fy) (fy)) => x

           (fmap! + (fx) (fy) (fy) (fy) [(fy)])
           => (throws UnsupportedOperationException))))

(defn test-fold-ge-matrix [factory]
  (let [x (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
        *' (fn ^double [^double x ^double y] (double (* x y)))
        +' (fn ^double [^double x ^double y] (double (+ x y)))]
    (facts "Fold implementation for real general matrix"
           (fold x) => 21.0
           (fold *' 1.0 x) => 720.0
           (fold +' 0.0 x) => (fold x))))

(defn test-reducible-ge-matrix [factory]
  (let [x (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
        y (create-ge-matrix factory 2 3 [2 3 4 5 6 7])
        pf1 (fn ^double [^double res ^double x] (+ x res))
        pf1o (fn [res ^double x] (conj res x))]
    (facts "Reducible implementation for real general matrix"

           (fold pf1 1.0 x) => 22.0
           (fold pf1o [] x) => [1.0 2.0 3.0 4.0 5.0 6.0]

           (fold pf1 1.0 x y) => 49.0
           (fold pf1o [] x y) => [3.0 5.0 7.0 9.0 11.0 13.0]

           (fold pf1 1.0 x y y) => 76.0
           (fold pf1o [] x y y) => [5.0 8.0 11.0 14.0 17.0 20.0]

           (fold + 1.0 x y y y) => (throws IllegalArgumentException))))

(defn test-all [factory]
  (do
    (test-equality factory)
    (test-release factory)
    (test-ifn-vector factory)
    (test-op-vector factory)
    (test-functor-vector factory)
    (test-fold-vector factory)
    (test-reducible-vector factory)
    (test-seq-vector factory)
    (test-ifn-ge-matrix factory)
    (test-op-ge-matrix factory)
    (test-functor-ge-matrix factory)
    (test-fold-ge-matrix factory)
    (test-reducible-ge-matrix factory)))
