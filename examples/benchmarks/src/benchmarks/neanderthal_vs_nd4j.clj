(ns benchmarks.neanderthal-vs-nd4j
  (:require [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer [mm! mm]]
             [native :refer [dge fge]]]
            [criterium.core :refer [quick-bench]])
  (:import org.nd4j.linalg.factory.Nd4j
           org.nd4j.linalg.api.ndarray.INDArray
           java.util.SplittableRandom))

(def m1 (Nd4j/rand 4 4))
(def m2 (Nd4j/rand 4 4))
(def result (Nd4j/createUninitialized (.shape ^INDArray m2)))
(.mmuli ^INDArray m1 ^INDArray m2 ^INDArray result)

(class (.data m1))

(defn bench-nd4j-mmuli-float [^long m ^long k ^long n]
  (let [m1 (Nd4j/rand m k)
        m2 (Nd4j/rand k n)
        result (Nd4j/createUninitialized (int-array [m n]) \f)]
    (quick-bench
     (do (.mmuli ^INDArray m1 ^INDArray m2 ^INDArray result)
         true))))

(defn bench-nd4j-gemm-float [^long m ^long k ^long n]
  (let [m1 (Nd4j/rand m k)
        m2 (Nd4j/rand k n)
        result (Nd4j/createUninitialized (int-array [m n]) \f)]
    (quick-bench
     (do (Nd4j/gemm ^INDArray m1 ^INDArray m2 ^INDArray result false false 1.0 0.0)
         true))))

(let [splittable-random (SplittableRandom.)]
  (defn random ^double [^double _]
    (.nextDouble ^SplittableRandom splittable-random)))

(defn bench-neanderthal-mm!-double [^long m ^long k ^long n]
  (with-release [m1 (fmap! random (dge m k))
                 m2 (fmap! random (dge k n))
                 result (dge m n)]
    (quick-bench
     (do (mm! 1.0 m1 m2 0.0 result)
         true))))

(defn bench-neanderthal-mm!-float [^long m ^long k ^long n]
  (with-release [m1 (fmap! random (fge m k))
                 m2 (fmap! random (fge k n))
                 result (fge m n)]
    (quick-bench
     (do (mm! 1.0 m1 m2 0.0 result)
         true))))

;; Vol 2

(defn bench-neanderthal-mm
  ([^long m ^long k ^long n]
   (with-release [m1 (fmap! random (fge m k))
                  m2 (fmap! random (fge k n))]
     (quick-bench
      (release (mm m1 m2)))))
  ([m k1 k2 k3 k4 k5 n]
   (with-release [m1 (fmap! random (fge m k1))
                  m2 (fmap! random (fge k1 k2))
                  m3 (fmap! random (fge k2 k3))
                  m4 (fmap! random (fge k3 k4))
                  m5 (fmap! random (fge k4 k5))
                  m6 (fmap! random (fge k5 n))]
     (quick-bench
      (release (mm m1 m2 m3 m4 m5 m6)))))
  ([dimensions]
   (with-release [ms (map (comp (partial fmap! random) fge)
                          (butlast dimensions) (rest dimensions))]
     (quick-bench
      (release (apply mm ms))))))

(defn bench-nd4j-mmul
  ([^long m ^long k ^long n]
   (let [m1 (Nd4j/rand m k)
         m2 (Nd4j/rand k n)]
     (quick-bench
      (do (.mmul ^INDArray m1 ^INDArray m2)
          true))))
  ([m k1 k2 k3 k4 k5 n]
   (let [m1 (Nd4j/rand ^int m ^int k1)
         m2 (Nd4j/rand ^int k1 ^int k2)
         m3 (Nd4j/rand ^int k2 ^int k3)
         m4 (Nd4j/rand ^int k3 ^int k4)
         m5 (Nd4j/rand ^int k4 ^int k5)
         m6 (Nd4j/rand ^int k5 ^int n)]
     (quick-bench
      (do (.mmul ^INDArray m1
                 (.mmul ^INDArray m2
                        (.mmul ^INDArray m3
                               (.mmul ^INDArray m4
                                      (.mmul ^INDArray m5 ^INDArray m6)) )))
          true)))))

(defn time-nd4j-mmul [m k1 k2 k3 k4 k5 n]
  (let [m1 (Nd4j/rand \f ^int m ^int k1)
        m2 (Nd4j/rand \f ^int k1 ^int k2)
        m3 (Nd4j/rand \f ^int k2 ^int k3)
        m4 (Nd4j/rand \f ^int k3 ^int k4)
        m5 (Nd4j/rand \f ^int k4 ^int k5)
        m6 (Nd4j/rand \f ^int k5 ^int n)]
    (time
     (do (.mmul ^INDArray m1
                (.mmul ^INDArray m2
                       (.mmul ^INDArray m3
                              (.mmul ^INDArray m4
                                     (.mmul ^INDArray m5 ^INDArray m6)) )))
         true))))

;; Vol 3

(defn bench-nd4j-mm-forwards [m k1 k2 k3 k4 k5 n]
  (let [m1 (Nd4j/rand ^int m ^int k1)
        m2 (Nd4j/rand ^int k1 ^int k2)
        m3 (Nd4j/rand ^int k2 ^int k3)
        m4 (Nd4j/rand ^int k3 ^int k4)
        m5 (Nd4j/rand ^int k4 ^int k5)
        m6 (Nd4j/rand ^int k5 ^int n)]
    (quick-bench
     (do (-> ^INDArray m1
             (.mmul ^INDArray m2)
             (.mmul ^INDArray m3)
             (.mmul ^INDArray m4)
             (.mmul ^INDArray m5)
             (.mmul ^INDArray m6))
         true))))

(defn time-nd4j-mm-forwards [m k1 k2 k3 k4 k5 n]
  (let [m1 (Nd4j/rand ^int m ^int k1)
        m2 (Nd4j/rand ^int k1 ^int k2)
        m3 (Nd4j/rand ^int k2 ^int k3)
        m4 (Nd4j/rand ^int k3 ^int k4)
        m5 (Nd4j/rand ^int k4 ^int k5)
        m6 (Nd4j/rand ^int k5 ^int n)]
    (time
     (do (-> ^INDArray m1
             (.mmul ^INDArray m2)
             (.mmul ^INDArray m3)
             (.mmul ^INDArray m4)
             (.mmul ^INDArray m5)
             (.mmul ^INDArray m6))
         true))))
