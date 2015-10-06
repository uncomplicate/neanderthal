(ns uncomplicate.neanderthal.real-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.clojurecl.core :refer [release with-release]]
            [uncomplicate.neanderthal
             [protocols :refer [data-accessor]]
             [core :refer :all]
             [math :refer :all]])
  (:import  [clojure.lang IFn$LLD IFn$LD IFn$DD]
            [uncomplicate.neanderthal.protocols RealBufferAccessor]))

(defn seq-to-buffer [^RealBufferAccessor acc s]
  (let [b (.createDataSource acc (count s))]
    (reduce (fn [^long i ^double e] (do (.set acc b i e) (inc i))) 0 s)
    b))

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

(defn test-group [factory]
  (facts "Group methods."
         (let [x (create-vector factory [1 2 3])
               y (create-vector factory [2 3 4])]

           (zero x) => (create-vector factory [0 0 0])
           (identical? x (zero x)) => false)

         (let [a (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
               ac (create-ge-matrix factory 2 3 [0 0 0 0 0 0])]
           (zero a) => ac
           (identical? a (zero a)) => false)))

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
        pf1o (fn [res ^double x] (conj res x))
        pf2 (fn ^double [^double res ^double x ^double y] (+ res x y))
        pf2o (fn [res ^double x ^double y] (conj res [x y]))
        pf3 (fn ^double [^double res ^double x ^double y ^double z]
              (+ res x y z))
        pf3o (fn [res ^double x ^double y ^double z] (conj res [x y z]))]
    (facts "Reducible implementation for vector"

           (freduce pf1 1.0 x) => 11.0
           (freduce pf1o [] x) => [1.0 2.0 3.0 4.0]

           (freduce pf2 1.0 x y) => 25.0
           (freduce pf2o [] x y)
           => [[1.0 2.0] [2.0 3.0] [3.0 4.0] [4.0 5.0]]

           (freduce pf3 1.0 x y y) => 39.0
           (freduce pf3o [] x y y)
           => [[1.0 2.0 2.0] [2.0 3.0 3.0] [3.0 4.0 4.0] [4.0 5.0 5.0]]

           (freduce + 1.0 x y y [y])
           => (throws UnsupportedOperationException))))

(defn test-seq-vector [factory]
  (facts "Vector as a sequence"
         (seq (create-vector factory [1 2 3])) => '(1.0 2.0 3.0)
         (seq (row (create-ge-matrix factory 2 3 (range 6)) 1)) => '(1.0 3.0 5.0)))

(defn test-ge-matrix [factory]
  (facts "Matrix methods."
         (let [a (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
               ac (create-ge-matrix factory 2 3 [0 0 0 0 0 0])]
           (mrows a) => 2
           (ncols a) => 3

           (row a 1) => (create-vector factory [2 4 6])

           (col a 1) => (create-vector factory [3 4]))))

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
        pf4 (fn ^double [^double x ^double y ^double z ^double w]
              (double (+ x y z w)))]
    (facts "Functor implementation for real general matrix"
           (instance? clojure.lang.IFn$DD pf1) => true

           (fmap! pf1 (fx)) => (fy)
           (fmap! pf1 x) => x

           (fmap! pf2 (fx) (fy)) => (create-ge-matrix factory 2 3 [3 5 7 9 11 13])
           (fmap! pf2 x (fy)) => x

           (fmap! pf3 (fx) (fy) (fy)) => (create-ge-matrix factory 2 3 [5 8 11 14 17 20])
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
        pf1o (fn [res ^double x] (conj res x))
        pf2 (fn ^double [^double res ^double x ^double y] (+ res x y))
        pf2o (fn [res ^double x ^double y] (conj res [x y]))
        pf3 (fn ^double [^double res ^double x ^double y ^double z]
              (+ res x y z))
        pf3o (fn [res ^double x ^double y ^double z]
               (conj res [x y z]))]
    (facts "Reducible implementation for real general matrix"

           (freduce pf1 1.0 x) => 22.0
           (freduce pf1o [] x) => [1.0 2.0 3.0 4.0 5.0 6.0]

           (freduce pf2 1.0 x y) => 49.0
           (freduce pf2o [] x y)
           => [[1.0 2.0] [2.0 3.0] [3.0 4.0] [4.0 5.0] [5.0 6.0] [6.0 7.0]]

           (freduce pf3 1.0 x y y) => 76.0
           (freduce pf3o [] x y y)
           => [[1.0 2.0 2.0] [2.0 3.0 3.0] [3.0 4.0 4.0] [4.0 5.0 5.0]
               [5.0 6.0 6.0] [6.0 7.0 7.0]]

           (freduce + 1.0 x y y [y])
           => (throws UnsupportedOperationException))))

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

;; ============= Matrices and Vectors ======================================

(defn test-vector-constructor [factory]
  (facts "Create a real vector."
         (vect? (create-vector factory [1 2 3])) => true
         (create-vector factory 1 2 3) => (create-vector factory [1 2 3])
         (create-vector factory 3) => (create-vector factory [0 0 0])
         (create-vector factory (seq-to-buffer (data-accessor factory) [1 2 3]))
         => (create-vector factory [1 2 3])
         (create-vector factory nil) => (throws IllegalArgumentException)
         (create-vector factory [])
         => (create-vector factory (seq-to-buffer (data-accessor factory) []))
         (create-vector factory 3) => (zero (create-vector factory [1 2 3]))))

(defn test-vector [factory]
  (facts "General vector functions."
         (dim (create-vector factory [1 2 3])) => 3
         (dim (create-vector factory [])) => 0

         (entry (create-vector factory [1 2 3]) 1) => 2.0
         (entry (create-vector factory []) 0)
         => (throws IndexOutOfBoundsException)

         (subvector (create-vector factory 1 2 3 4 5 6) 1 3)
         => (create-vector factory 2 3 4)
         (subvector (create-vector factory 1 2 3) 2 3)
         => (throws IndexOutOfBoundsException)))

;;================ BLAS functions =========================================

(defn test-dot [factory]
  (facts "BLAS 1 dot."
         (dot (create-vector factory [1 2 3]) (create-vector factory [10 30 -10]))
         => 40.0
         (dot (create-vector factory [1 2 3]) nil)
         => (throws IllegalArgumentException)
         (dot (create-vector factory []) (create-vector factory [])) => 0.0
         (dot (create-vector factory [1]) (create-ge-matrix factory 3 3))
         => (throws IllegalArgumentException)
         (dot (create-vector factory [1 2]) (create-vector factory [1 2 3]))
         => (throws IllegalArgumentException)))

(defn test-nrm2 [factory]
  (facts "BLAS 1 nrm2."
         (nrm2 (create-vector factory [1 2 3]))  => (roughly (Math/sqrt 14))
         (nrm2 (create-vector factory [])) => 0.0))

(defn test-asum [factory]
  (facts "BLAS 1 asum."
         (asum (create-vector factory [1 2 -3]))  => 6.0
         (asum (create-vector factory [])) => 0.0))

(defn test-sum [factory]
  (facts "BLAS Plus 1 sum."
         (sum (create-vector factory [1 2 -5]))  => -2.0
         (sum (create-vector factory [])) => 0.0))

(defn test-iamax [factory]
  (facts "BLAS 1 iamax"
         (iamax (create-vector factory [1 2 7 7 6 2 -10 10])) => 6
         (iamax (create-vector factory [])) => 0
         (iamax (create-vector factory [4 6 7 3])) => 2))

(defn test-rot [factory]
  (facts "BLAS 1 rot!"
         (let [x (create-vector factory [1 2 3 4 5])
               y (create-vector factory [-1 -2 -3 -4 -5])]
           (do (rot! x y 1 0) [x y])
           => [(create-vector factory 1 2 3 4 5)
               (create-vector factory -1 -2 -3 -4 -5)]

           (do (rot! x y 0.5 (/ (sqrt 3) 2.0)) [x y]
               (+ (double (nrm2 (axpy -1 x (create-vector factory
                                                          -0.3660254037844386
                                                          -0.7320508075688772
                                                          -1.098076211353316
                                                          -1.4641016151377544
                                                          -1.8301270189221928))))
                  (double (nrm2 (axpy -1 y (create-vector factory
                                                          -1.3660254037844386
                                                          -2.732050807568877
                                                          -4.098076211353316
                                                          -5.464101615137754
                                                          -6.830127018922193))))))
           => (roughly 0 1e-6)

           (rot! x y (/ (sqrt 3) 2.0)  0.5) => x
           (rot! x (create-vector factory 1) 0.5 0.5)
           => (throws IllegalArgumentException)
           (rot! x y 300 0.3) => (throws IllegalArgumentException)
           (rot! x y 0.5 0.5) => (throws IllegalArgumentException))))

(defn test-rotg [factory]
  (facts "BLAS 1 rotg!"
         (let [x (create-vector factory 1 2 3 4)]
           (rotg! x) => x)
         (rotg! (create-vector factory 0 0 0 0))
         => (create-vector factory 0 0 1 0)
         (rotg! (create-vector factory 0 2 0 0))
         => (create-vector factory 2 1 0 1)
         (nrm2 (axpy -1 (rotg! (create-vector factory 6 -8 0 0))
                     (create-vector factory -10 -5/3 -3/5 4/5)))
         => (roughly 0 1e-6)))

(defn test-rotm [factory]
  (facts "BLAS 1 rotm!"
         "TODO: Leave this for later since I am lacking good examples right now."))

(defn test-rotmg [factory]
  (facts "BLAS 1 rotmg!"
         "TODO: Leave this for later since I am lacking good examples right now."))

(defn test-swap [factory]
  (facts
   "BLAS 1 swap! vectors"
   (let [x (create-vector factory [1 2 3])]
     (swp! x (create-vector factory 3)) => x
     (swp! x (create-vector factory [10 20 30]))
     => (create-vector factory [10 20 30])
     (swp! x nil) => (throws IllegalArgumentException)
     (identical? (swp! x (create-vector factory [10 20 30])) x) => true
     (swp! x (create-vector factory [10 20]))
     => (throws IllegalArgumentException)))

  (facts
   "BLAS 1 swap! matrices"
   (let [a (create-ge-matrix factory 2 3 [1 2 3 4 5 6])]
     (swp! a (create-ge-matrix factory 2 3)) => a
     (swp! a (create-ge-matrix factory 2 3 [10 20 30 40 50 60]))
     => (create-ge-matrix factory 2 3 [10 20 30 40 50 60])
     (swp! a nil) => (throws IllegalArgumentException)
     (identical? (swp! a (create-ge-matrix factory 2 3 [10 20 30 40 50 60])) a)
     => true
     (swp! a (create-ge-matrix factory 1 2 [10 20]))
     => (throws IllegalArgumentException))))

(defn test-scal [factory]
  (facts "BLAS 1 scal!"
         (let [x (create-vector factory [1 2 3])]()
              (identical? (scal! 3 x) x) => true)
         (scal! 3 (create-vector factory [1 -2 3]))
         => (create-vector factory [3 -6 9])
         (scal! 3 (create-vector factory []))
         => (create-vector factory [])))

(defn test-copy [factory]
  (facts "BLAS 1 copy!"
         (let [y (create-vector factory 3)]
           (copy! (create-vector factory [1 2 3]) y) => y
           (copy (create-vector factory [1 2 3])) => y)

         (copy! (create-vector factory [10 20 30])
                (create-vector factory [1 2 3]))
         => (create-vector factory [10 20 30])
         (copy! (create-vector factory [1 2 3]) nil)
         => (throws IllegalArgumentException)
         (copy! (create-vector factory [10 20 30]) (create-vector factory [1]))
         => (throws IllegalArgumentException))

  (facts "BLAS 1 copy"
         (copy (create-vector factory 1 2)) => (create-vector factory 1 2)
         (let [x (create-vector factory 1 2)] (identical? (copy x) x) => false)))

(defn test-axpy [factory]
  (facts "BLAS 1 axpy!"
         (let [y (create-vector factory [1 2 3])]
           (axpy! 2.0 (create-vector factory [10 20 30]) y)
           => (create-vector factory [21 42 63])
           (identical? (axpy! 3.0 (create-vector factory [1 2 3]) y) y) => true
           (axpy! 2.0 (create-vector factory [1 2]) y)
           => (throws IllegalArgumentException)
           (axpy! (create-vector factory [10 20 30])
                  (create-vector factory [1 2 3]))
           => (create-vector factory [11 22 33]))

         (axpy! 2 (create-vector factory 1 2 3) (create-vector factory 1 2 3)
                (create-vector factory 2 3 4) 4.0 (create-vector factory 5 6 7))
         => (create-vector factory 25 33 41)
         (axpy! 2 (create-vector factory 1 2 3) (create-vector factory 1 2 3)
                (create-vector factory 2 3 4) 4.0)
         => (throws IllegalArgumentException)
         (axpy! (create-vector factory 1 2 3) (create-vector factory 1 2 3)
                (create-vector factory 1 2 3) (create-vector factory 1 2 3)
                (create-vector factory 1 2 3))
         => (axpy! 5 (create-vector factory 1 2 3) (create-vector factory 3))
         (axpy! 4 "af" (create-vector factory 1 2 3) 3 "b" "c")
         => (throws IllegalArgumentException)
         (let [y (create-vector factory [1 2 3])]
           (axpy! 2 (create-vector factory 1 2 3) y
                  (create-vector factory 2 3 4) 4.0 (create-vector factory 5 6 7))
           => y))

  (facts "BLAS1 axpy"
         (let [y (create-vector factory [1 2 3])]
           (axpy 2.0 (create-vector factory [10 20 30]))
           => (throws IllegalArgumentException)
           (identical? (axpy 3.0 (create-vector factory [1 2 3]) y) y) => false
           (axpy (create-vector factory 1 2 3) (create-vector factory 2 3 4)
                 (create-vector factory 3 4 5) (create-vector factory 3))
           => (create-vector factory 6 9 12)
           (axpy 2 (create-vector factory 1 2 3) (create-vector factory 2 3 4))
           => (create-vector factory 4 7 10))))

;; ================= Real General Matrix functions =====================================
(defn test-matrix-constructor [factory]
  (facts "Create a matrix."
         (let [a (create-ge-matrix factory 2 3 [1 2 3 4 5 6])]
           (create-ge-matrix factory 2 3 [1 2 3 4 5 6]) => a
           (create-ge-matrix factory 2 3 (seq-to-buffer (data-accessor factory)
                                                        [1 2 3 4 5 6]))
           => a
           (create-ge-matrix factory 2 3 nil) => (throws IllegalArgumentException)
           (create-ge-matrix factory 0 0 [])
           => (create-ge-matrix factory 0 0 (seq-to-buffer (data-accessor factory) [])))))

(defn test-matrix [factory]
  (facts "General Matrix methods."
         (ncols (create-ge-matrix factory 2 3 [1 2 3 4 5 6])) => 3
         (ncols (create-ge-matrix factory 0 0 [])) => 0

         (mrows (create-ge-matrix factory 2 3)) => 2
         (mrows (create-ge-matrix factory 0 0)) => 0

         (entry (create-ge-matrix factory 2 3 [1 2 3 4 5 6]) 0 1) => 3.0
         (entry (create-ge-matrix factory 0 0) 0 0)
         => (throws IndexOutOfBoundsException)
         (entry (create-ge-matrix factory 2 3 [1 2 3 4 5 6]) -1 2)
         => (throws IndexOutOfBoundsException)
         (entry (create-ge-matrix factory 2 3 [1 2 3 4 5 6]) 2 1)
         => (throws IndexOutOfBoundsException)

         (row (create-ge-matrix factory 2 3 [1 2 3 4 5 6]) 1)
         => (create-vector factory 2 4 6)
         (row (create-ge-matrix factory 2 3 [1 2 3 4 5 6]) 4)
         => (throws IndexOutOfBoundsException)

         (col (create-ge-matrix factory 2 3 [1 2 3 4 5 6]) 1)
         => (create-vector factory 3 4)
         (col (create-ge-matrix factory 2 3 [1 2 3 4 5 6]) 4)
         => (throws IndexOutOfBoundsException)

         (submatrix (create-ge-matrix factory 3 4 (range 12)) 1 2 2 1)
         => (create-ge-matrix factory 2 1 [7 8])
         (submatrix (create-ge-matrix factory 3 4 (range 12)) 2 3)
         => (create-ge-matrix factory 2 3 [0 1 3 4 6 7])
         (submatrix (create-ge-matrix factory 3 4 (range 12)) 3 4)
         => (create-ge-matrix factory 3 4 (range 12))
         (submatrix (create-ge-matrix factory 3 4 (range 12)) 1 1 3 3)
         => (throws IndexOutOfBoundsException)))

;; ====================== BLAS 2 ===============================

(defn test-mv [factory]
  (facts "BLAS 2 mv!"
         (mv! 2.0 (create-ge-matrix factory 3 2 [1 2 3 4 5 6])
              (create-vector factory 1 2 3) 3 (create-vector factory [1 2 3 4]))
         => (throws IllegalArgumentException)

         (let [y (create-vector factory [1 2 3])]
           (mv! 2 (create-ge-matrix factory 3 2 [1 2 3 4 5 6])
                (create-vector factory 1 2) 3 y)
           => y)

         (mv! (create-ge-matrix factory 2 3 [1 10 2 20 3 30])
              (create-vector factory 7 0 4) (create-vector factory 0 0))
         => (create-vector factory 19 190)

         (mv! 2.0 (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
              (create-vector factory 1 2 3) 3.0 (create-vector factory 1 2))
         => (create-vector factory 47 62)

         (mv! 2.0 (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
              (create-vector factory 1 2 3) 0.0 (create-vector factory 0 0))
         => (mv 2.0 (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
                (create-vector factory 1 2 3))

         (mv! (trans (create-ge-matrix factory 3 2 [1 3 5 2 4 6]))
              (create-vector factory 1 2 3) (create-vector factory 0 0))
         => (mv! (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
                 (create-vector factory 1 2 3) (create-vector factory 0 0))))

(defn test-rank [factory]
  (facts "BLAS 2 rank!"
         (let [a (create-ge-matrix factory 2 3 [1 2 3 4 5 6])]
           (rank! 2.0 (create-vector factory 1 2)
                  (create-vector factory 1 2 3) a)
           => a)
         (rank! (create-vector factory 3 2 1 4)
                (create-vector factory 1 2 3)
                (create-ge-matrix factory 4 3 [1 2 3 4 2 2 2 2 3 4 2 1]))
         => (create-ge-matrix factory 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

         (rank! (create-vector factory 1 2) (create-vector factory 1 2 3)
                (create-ge-matrix factory 2 2 [1 2 3 5]))
         => (throws IllegalArgumentException)))

;; ====================== BLAS 3 ================================

(defn test-mm [factory]
  (facts "BLAS 3 mm!"
         (mm! 2.0 (create-ge-matrix factory 3 2 [1 2 3 4 5 6])
              (create-ge-matrix factory 3 2 [1 2 3 4 5 6])
              3.0 (create-ge-matrix factory 3 2 [1 2 3 4 5 6]))
         => (throws IllegalArgumentException)

         (mm! 2.0 (create-ge-matrix factory 3 2 [1 2 3 4 5 6])
              (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
              3.0 (create-ge-matrix factory 3 2 [1 2 3 4 5 6]))
         => (throws IllegalArgumentException)

         (let [c (create-ge-matrix factory 2 2 [1 2 3 4])]
           (mm! 2.0 (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
                (create-ge-matrix factory 3 2 [1 2 3 4 5 6]) 3.0 c)
           => c)

         (mm! 2.0 (create-ge-matrix factory 2 3 [1 2 3 4 5 6])
              (create-ge-matrix factory 3 2 [1 2 3 4 5 6])
              3.0 (create-ge-matrix factory 2 2 [1 2 3 4]))
         => (create-ge-matrix factory 2 2 [47 62 107 140])

         (mm! 2.0 (trans (create-ge-matrix factory 3 2 [1 3 5 2 4 6]))
              (trans (create-ge-matrix factory 2 3 [1 4 2 5 3 6]))
              3.0  (create-ge-matrix factory 2 2 [1 2 3 4]))
         => (create-ge-matrix factory 2 2 [47 62 107 140]))

  (facts "mm"
         (mm (create-ge-matrix factory 3 2 (range 6))
             (create-ge-matrix factory 2 3 (range 6)))
         => (create-ge-matrix factory 3 3 [3 4 5 9 14 19 15 24 33])))

(defn test-all [factory]
  (do (test-equality factory)
      (test-group factory)
      (test-ifn-vector factory)
      (test-functor-vector factory)
      (test-fold-vector factory)
      (test-reducible-vector factory)
      (test-seq-vector factory)
      (test-ge-matrix factory)
      (test-ifn-ge-matrix factory)
      (test-functor-ge-matrix factory)
      (test-fold-ge-matrix factory)
      (test-reducible-ge-matrix factory)
      (test-release factory)
      (test-vector-constructor factory)
      (test-vector factory)
      (test-dot factory)
      (test-nrm2 factory)
      (test-asum factory)
      (test-sum factory)
      (test-iamax factory)
      (test-rot factory)
      (test-rotg factory)
      (test-rotm factory)
      (test-rotmg factory)
      (test-swap factory)
      (test-scal factory)
      (test-copy factory)
      (test-axpy factory)
      (test-matrix-constructor factory)
      (test-matrix factory)
      (test-mv factory)
      (test-rank factory)
      (test-mm factory)))
