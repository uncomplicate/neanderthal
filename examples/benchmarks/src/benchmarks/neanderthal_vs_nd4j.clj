(ns benchmarks.neanderthal-vs-nd4j
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer [mm!]]
             [native :refer [dge fge]]]
            [criterium.core :refer [quick-bench]])
  (:import org.nd4j.linalg.factory.Nd4j
           org.nd4j.linalg.api.ndarray.INDArray
           org.nd4j.linalg.cpu.nativecpu.NDArray
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
