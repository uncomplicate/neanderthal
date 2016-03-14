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

(def nx (dv (range n)))
(def ny (dv (range n)))

(with-progress-reporting (quick-bench (dot nx ny)))
;; Execution time mean : 32.828791 µs

(defn p+ ^double [^double x ^double y]
  (+ x y))

(defn p* ^double [^double x ^double y]
  (* x y))

(defn sqr ^double [^double x]
  (* x x))

(with-progress-reporting (quick-bench (foldmap p+ 0.0 p* cvx cvy)))
;; Execution time mean : 2.316850 ms

(with-progress-reporting (quick-bench (foldmap p+ 0.0 p* nx ny)))
;; Execution time mean : 186.881027 µs

(with-progress-reporting (quick-bench (fold nx)))
;; Execution time mean : 95.081694 µs

(with-progress-reporting
  (quick-bench
   (sqrt (- (/ (foldmap p+ 0.0 sqr nx) n) (pow (/ (fold nx) n) 2)))))
;; Execution time mean : 197.692695 µs
