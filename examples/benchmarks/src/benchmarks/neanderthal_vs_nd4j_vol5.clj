(ns benchmarks.neanderthal-vs-nd4j-vol5
  (:require [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.clojurecuda.core
             :refer [init context device current-context current-context! synchronize!]]
            [uncomplicate.clojurecl.core :refer [set-default! release-context!]]
            [uncomplicate.neanderthal
             [core :refer [dot raw row col entry! mv! sum mrows asum axpy! vctr rk! iamax]]
             [native :refer [fge fv]]
             [cuda :refer [cuge cuv set-engine!]]
             [opencl :refer [clge clv] :as opencl]
             [math :refer [sqrt]]]
            [criterium.core :refer [quick-bench]])
  (:import org.nd4j.linalg.factory.Nd4j
           org.nd4j.linalg.api.ndarray.INDArray
           java.util.SplittableRandom))

(let [splittable-random (SplittableRandom.)]
  (defn random ^double [^double _]
    (.nextDouble ^SplittableRandom splittable-random)))

(defn bench-nd4j-distance
  ([n]
   (let [x (Nd4j/rand (int-array [n]))
         y (Nd4j/rand (int-array [n]))]
     (quick-bench (.distance2 ^INDArray x ^INDArray y))))
  ([m n]
   (let [x (Nd4j/rand \f (int-array [m n]))
         y (Nd4j/rand \f (int-array [m n]))]
     (quick-bench (.distance2 ^INDArray x ^INDArray y)))))

;; 2: 2.865009 µs
;; 10: 3.055880 µs
;; 100: 3.104767 µs
;; 1000: 3.712104 µs
;; 10000: 7.369057 µs
;; 1000000: 369.138225 µs
;; 100000000: 39.677566 ms

;; 10x15: 3.313088 µs
;; 100x100: 7.109816 µs
;; 1000x1000: 357.002435 µs
;; 100x10000: 423.156695 µs
;; 10000x100: 335.000853 µs
;; 10000x10000: 40.107896 ms

(defn bench-neanderthal-distance
  ([n]
   (with-release [x (fmap! random (fv n))
                  y (fmap! random (fv n))]
     (quick-bench (sqrt (dot x y)))))
  ([m n]
   (with-release [x (fmap! random (fge m n))
                  y (fmap! random (fge m n))]
     (quick-bench (sqrt (dot x y))))))

;; 2: 64.005593 ns
;; 10: 62.042329 ns
;; 100: 66.816429 ns
;; 1000: 92.034678 ns
;; 10000: 908.968553 ns
;; 1000000: 68.781036 µs
;; 100000000: 25.768197 ms

;; 10x15: 278.192235 ns
;; 100x100: 1.115164 µs
;; 1000x1000: 82.225425 µs
;; 100x10000: 76.504216 µs
;; 10000x100: 81.241423 µs
;; 10000x10000: 27.167233 ms

(defn bench-nd4j-sum
  ([n]
   (let [x (Nd4j/rand (int-array [n]))]
     (quick-bench (.sumNumber ^INDArray x))))
  ([m n]
   (let [a (Nd4j/rand \f (int-array [m n]))
         d (int-array [0])]
     (quick-bench (.sum ^INDArray a d)))))

;; 2: 4.214011 µs
;; 10: 4.232423 µs
;; 100: 4.299567 µs
;; 1000: 4.627851 µs
;; 10000: 14.572367 µs
;; 1000000: 219.302064 µs
;; 100000000: 22.361447 ms

;; 10x15: 5.904680 µs
;; 100x100: 21.092154 µs
;; 1000x1000: 52.358064 µs
;; 100x10000: 103.955403 µs
;; 10000x100: 64.909655 µs
;; 10000x10000: 15.360876 ms

;;CUDA:
;; 2:
;; 10:
;; 100:
;; 1000:
;; 10000:
;; 1000000: 423.446584 µs
;; 100000000: 5.578132 ms

(defn bench-neanderthal-sum
  ([n]
   (with-release [x (fmap! random (fv n))]
     (quick-bench (sum x))))
  ([m n]
   (with-release [a (fmap! random (fge m n))]
     (quick-bench (sum a)))))

;; 2: 71.435637 ns
;; 10: 70.562578 ns
;; 100: 84.437461 ns
;; 1000: 120.773336 ns
;; 10000: 1.225827 µs
;; 1000000: 130.245611 µs
;; 100000000: 28.147435 ms

;; 10x15: 84.925045 ns
;; 100x100: 1.254015 µs
;; 1000x1000: 131.412269 µs
;; 100x10000: 147.992786 µs
;; 10000x100: 129.061267 µs
;; 10000x10000: 27.866048 ms

(defn bench-neanderthal-asum
  ([n]
   (with-release [x (fmap! random (fv n))]
     (quick-bench (asum x))))
  ([m n]
   (with-release [a (fmap! random (fge m n))]
     (quick-bench (asum a)))))

;; 2: 50.302131 ns
;; 10: 45.395512 ns
;; 100: 49.682850 ns
;; 1000: 76.036001 ns
;; 10000: 573.208513 ns
;; 1000000: 32.646665 µs
;; 100000000: 13.238496 ms

;; 10x15: 50.816077 ns
;; 100x100: 466.800109 ns
;; 1000x1000: 31.799570 µs
;; 100x10000: 33.643302 µs
;; 10000x100: 32.994745 µs
;; 10000x10000: 13.070991 ms

(defn bench-neanderthal-cuda-sum
  ([n]
   (with-release [x (cuv (fmap! random (fv n)))]
     (quick-bench (sum x))))
  ([m n]
   (with-release [a (cuge (fmap! random (fge m n)))]
     (quick-bench (sum a)))))

;; 2: 29.293088 µs
;; 10: 29.786383 µs
;; 100: 29.782401 µs
;; 1000: 28.673259 µs
;; 10000: 32.925326 µs
;; 1000000: 276.082162 µs
;; 100000000: 4.847634 ms
;; asum is roughly 4x faster for huge sizes

;; 10x15: 30.058548 µs
;; 100x100: 32.334560 µs
;; 1000x1000: 281.033320 µs
;; 100x10000: 272.578388 µs
;; 10000x100: 272.715423 µs
;; 10000x10000: 4.330174 ms

(defn bench-neanderthal-cuda-asum
  ([n]
   (with-release [x (cuv (fmap! random (fv n)))]
     (quick-bench (asum x))))
  ([m n]
   (with-release [a (cuge (fmap! random (fge m n)))]
     (quick-bench (asum a)))))

(init)
(current-context! (context (device 0)))
(set-engine!)

;; 100000000: 1.272327 ms

(release (current-context))

(set-default!)
(opencl/set-engine!)

(defn bench-neanderthal-opencl-sum
  ([n]
   (with-release [x (clv (fmap! random (fv n)))]
     (quick-bench (sum x))))
  ([m n]
   (with-release [a (clge (fmap! random (fge m n)))]
     (quick-bench (sum a)))))

;; 2: 62.114993 µs
;; 10: 68.475910 µs
;; 100: 63.432086 µs
;; 1000: 63.189027 µs
;; 10000: 62.111563 µs
;; 1000000: 75.166088 µs
;; 100000000: 1.532834 ms

;; 10x15:
;; 100x100:
;; 1000x1000: 76.703746 µs
;; 100x10000:
;; 10000x100:
;; 10000x10000: 1.540860 ms

(release-context!)

(defn bench-nd4j-iamax
  [n]
  (let [x (Nd4j/rand (int-array [n]))
        exec (Nd4j/getExecutioner)
        iamax (org.nd4j.linalg.api.ops.impl.indexaccum.IAMax. x)]
    (quick-bench (.getFinalResult (.execAndReturn exec iamax)))))

;; 2: 1.212382 µs
;; 10: 1.234349 µs
;; 100: 1.346079 µs
;; 1000: 2.080715 µs
;; 10000: 10.239278 µs
;; 1000000: 209.913386 µs
;; 100000000: 20.508549 ms

(defn bench-neanderthal-iamax [n]
  (with-release [x (fmap! random (fv n))]
    (quick-bench (iamax x))))

;; 2: 75.503383 ns
;; 10: 66.782601 ns
;; 100: 63.711496 ns
;; 1000: 305.853934 ns
;; 10000: 2.540639 µs
;; 1000000: 116.195962 µs
;; 100000000: 15.084416 ms
