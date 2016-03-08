(ns uncomplicate.neanderthal.buffer-block-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.clojurecl.core :refer [release with-release]]
            [uncomplicate.fluokitten.core :refer [fmap! fold foldmap]]
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

(defn test-functor-vector [factory]
  (let [fx (fn [] (create-vector factory [1 2 3 4]))
        fy (fn [] (create-vector factory [2 3 4 5 6]))
        x (fx)
        pf1 (fn ^double [^double x] (+ x 1.0))
        pf2 (fn ^double [^double x ^double y] (+ x y))
        pf3 (fn ^double [^double x ^double y ^double z] (+ x y z))
        pf4 (fn ^double [^double x ^double y ^double z ^double w]
              (+ x y z w))]
    (facts "Functor implementation for real block vector"
           (instance? IFn$DD pf1) => true

           (fmap! pf1 (fx)) => (create-vector factory [2 3 4 5])
           (fmap! pf1 x) => x

           (fmap! pf2 (fx) (fy)) => (create-vector factory [3 5 7 9])
           (fmap! pf2 x (fy)) => x

           (fmap! pf3 (fx) (fy) (fy)) => (create-vector factory [5 8 11 14])
           (fmap! pf3 x (fy) (fy)) => x

           (fmap! pf4 (fx) (fy) (fy) (fy)) => (create-vector factory [7 11 15 19])
           (fmap! pf4 x (fy) (fy) (fy)) => x

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

(defn test-functor-ge-matrix [factory]
  (let [fx (fn [] (create-ge-matrix factory 2 3 [1 2 3 4 5 6]))
        fy (fn [] (create-ge-matrix factory 2 3 [2 3 4 5 6 7]))
        x (fx)
        pf1 (fn ^double [^double x] (double (+ x 1.0)))
        pf2 (fn ^double [^double x ^double y] (double (+ x y)))
        pf3 (fn ^double [^double x ^double y ^double z] (double (+ x y z)))
        pf4 (fn ^double [^double x ^double y ^double z ^double v]
              (double (+ x y z v)))]
    (facts "Functor implementation for real general matrix"
           (instance? clojure.lang.IFn$DD pf1) => true

           (fmap! pf1 (fx)) => (fy)
           (fmap! pf1 x) => x

           (fmap! pf2 (fx) (fy))
           => (create-ge-matrix factory 2 3 [3 5 7 9 11 13])
           (fmap! pf2 x (fy)) => x

           (fmap! pf3 (fx) (fy) (fy))
           => (create-ge-matrix factory 2 3 [5 8 11 14 17 20])
           (fmap! pf3 x (fy) (fy)) => x

           (fmap! pf4 (fx) (fy) (fy) (fy))
           => (create-ge-matrix factory 2 3 [7 11 15 19 23 27])
           (fmap! pf4 x (fy) (fy) (fy)) => x

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
    (test-functor-vector factory)
    (test-fold-vector factory)
    (test-reducible-vector factory)
    (test-seq-vector factory)
    (test-ifn-ge-matrix factory)
    (test-functor-ge-matrix factory)
    (test-fold-ge-matrix factory)
    (test-reducible-ge-matrix factory)))
