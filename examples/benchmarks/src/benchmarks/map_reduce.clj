(ns benchmarks.map-reduce
  (:require [uncomplicate.fluokitten.core :refer [fmap! fmap foldmap fold]]
            [uncomplicate.neanderthal
             [core :refer :all :exclude [entry! entry]]
             [real :refer [entry! entry]]
             [native :refer :all]
             [math :refer :all]]
            [hiphip.double :as hiphip]
            [criterium.core :refer :all]))

(def n 100000)

;; =================== Clojure vectors ========================

(def cvx (vec (range n)))
(def cvy (vec (range n)))

(with-progress-reporting (quick-bench (reduce + (map * cvx cvy))))
;; Execution time mean : 11.493054 ms

(with-progress-reporting (quick-bench (reduce + cvx)))
;; Execution time mean : 970.238233 µs

;; =================== Primitive arrays ===============

(def cax (double-array (range n)))
(def cay (double-array (range n)))

(with-progress-reporting
  (quick-bench
   (areduce ^doubles cax i ret 0.0
            (+ ret (* (aget ^doubles cax i)
                      (aget ^doubles cay i))))))
;; Execution time mean : 141.939897 µs

(with-progress-reporting (quick-bench (hiphip/asum [x cax y cay] (* x y))))
;; Execution time mean : 91.268532 µs

(with-progress-reporting
  (quick-bench
   (areduce ^doubles cax i ret 0.0
            (+ ret (aget ^doubles cax i)))))
;; Execution time mean : 92.133214 µs

(with-progress-reporting
  (quick-bench
   (sqr (- (/ (hiphip/asum [x cax] (* x x)) n) (pow (/ (hiphip/asum [x cax] x)) 2)))))
;; Execution time mean : 274.978697 µs

;; ==================== Fluokitten & Neanderthal ===================

(defn p+ ^double [^double x ^double y]
  (+ x y))

(defn p* ^double [^double x ^double y]
  (* x y))

(with-progress-reporting (quick-bench (foldmap p+ 0.0 p* cax cay)))
;; Execution time mean : 80.978520 µs

(def nx (dv (range n)))
(def ny (dv (range n)))

(with-progress-reporting (quick-bench (dot nx ny)))
;; Execution time mean : 8.472175 µs

(with-progress-reporting (quick-bench (foldmap p+ 0.0 p* cvx cvy)))
;; Execution time mean : 2.595561 ms

(with-progress-reporting (quick-bench (foldmap p+ 0.0 p* nx ny)))
;; Execution time mean : 198.744652 µs

(with-progress-reporting (quick-bench (fold nx)))
;; Execution time mean : 20.372299 µs

(with-progress-reporting
  (quick-bench
   (sqrt (- (/ (foldmap sqr nx) n) (pow (/ (fold nx) n) 2)))))
;; Execution time mean : 129.750104 µs

(def mx (dge 316 316 (range n)))
(def my (dge 316 316 (range n)))

(with-progress-reporting (quick-bench (foldmap p+ 0.0 p* mx my)))
;; Execution time mean : 210.489030 µs
