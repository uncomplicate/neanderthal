(ns benchmarks.neanderthal-vs-nd4j-vol4
  (:require [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.clojurecuda.core
             :refer [init context device current-context current-context! synchronize!]]
            [uncomplicate.neanderthal
             [core :refer [dot raw row col entry! mrows rows asum axpy! vctr rk!]]
             [native :refer [fge fv]]
             [cuda :refer [cuge cuv set-engine!]]
             [math :refer [sqrt]]]
            [criterium.core :refer [quick-bench]])
  (:import org.nd4j.linalg.factory.Nd4j
           org.nd4j.linalg.api.ndarray.INDArray))

;; Vol 4

(defn bench-nd4j-addiRowVector [^long m ^long n]
  (let [a (Nd4j/ones m n)
        x (Nd4j/ones n)]
    (quick-bench
     (do (.addiRowVector ^INDArray a ^INDArray x)
         true))))

;; 10x15: 1.687800 µs
;; 100x100: 7.704195 µs
;; 1000x1000: 67.836326 µs
;; 100x10000: 67.930071 µs
;; 10000x100: 83.789454 µs
;; 10000x10000: 29.833284 ms

(defn bench-nd4j-addiRowVector-cuda [^long m ^long n]
  (let [a (Nd4j/ones m n)
        x (Nd4j/ones n)]
    (quick-bench
     (do (.addiRowVector ^INDArray a ^INDArray x)
         (.sumNumber ^INDArray x)))))

;; Nd4j on CUDA:
;; 10x15: 241.076577 µs
;; 100x100: 138.564805 µs
;; 1000x1000: 223.194717 µs
;; 100x10000: 289.878918 µs
;; 10000x100: 211.342593 µs
;; 10000x10000: 7.895556 ms

(defn naive-add-row-vector! [a x]
  (doseq [rw (rows a)]
    (axpy! x rw))
  a)

(defn bench-neanderthal-naive-add-row-vector [m n]
  (with-release [a (entry! (fge m n {:layout :row}) 1)
                 x (entry! (fv n) 1)]
    (quick-bench
     (do (naive-add-row-vector! a x)
         true))))

;; 10x15: 1.379596 µs
;; 100x100: 11.467549 µs
;; 1000x1000: 216.736544 µs
;; 100x10000: 140.238912 µs
;; 10000x100: 1.163332 ms
;; 10000x10000: 31.391642 ms

(defn less-naive-add-row-vector! [a x]
  (with-release [y (entry! (raw (col a 0)) 1)]
    (rk! y x a)))

(defn bench-neanderthal-less-naive-add-row-vector [m n]
  (with-release [a (entry! (fge m n {:layout :row}) 1)
                 x (entry! (fv n) 1)]
    (quick-bench
     (do (less-naive-add-row-vector! a x)
         true))))

;; 10x15: 506.063330 ns
;; 100x100: 1.768599 µs
;; 1000x1000: 61.031548 µs
;; 100x10000: 155.047204 µs
;; 10000x100: 66.700431 µs
;; 10000x10000: 26.320592 ms

(defn bench-neanderthal-less-naive-add-row-vector-cuda [m n]
  (with-release [a (entry! (cuge m n) 1)
                 x (entry! (cuv n) 1)]
    (quick-bench
     (do (less-naive-add-row-vector! a x)
         (asum x)))))

(init)
(current-context! (context (device 0)))
(set-engine!)
(release (current-context))

;; 10x15: 52.111330 µs
;; 100x100: 52.281520 µs
;; 1000x1000: 81.112859 µs
;; 100x10000: 80.279229 µs
;; 10000x100: 74.671313 µs
;; 10000x10000: 2.601956 ms
