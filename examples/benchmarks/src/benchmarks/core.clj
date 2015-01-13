(ns benchmarks.core
  (:require [uncomplicate.neanderthal
             [core :refer :all]
             [real :refer :all]
             [math :refer :all]]
            [clojure.core.matrix :as cm]
            [clojure.core.matrix.operators :as cmop]
            [criterium.core :as criterium])
  (:import [org.jblas DoubleMatrix]
           [mikera.matrixx Matrix]
           [mikera.matrixx.algo Multiplications]))

(defn rnd ^double [^double x]
  (Math/random))

(defn benchmark-neanderthal-mm [n]
  (let [as (map #(fmap! rnd (dge (pow 2 %) (pow 2 %))) (range 1 n))
        bs (map #(fmap! rnd (dge (pow 2 %) (pow 2 %))) (range 1 n))
        cs (map #(dge (pow 2 %) (pow 2 %)) (range 1 n))]
    (map #((:sample-mean (criterium/quick-benchmark (mm! %1 1.0 %2 %3 0.0) {})) 0) cs as bs)))

(defn benchmark-array-sum [n]
  (let [arrays (map #(double-array (* (pow 2 %) (* (pow 2 %)))) (range 1 n))
        sum (fn [a]
              (let [len (alength a)]
                (loop [i 0 sum 0]
                  (if (< i len)
                    (recur (inc i) (+ sum (aget a i)))
                    sum))))]
    (map #((:sample-mean (criterium/quick-benchmark (sum %) {})) 0) arrays)))

(defn benchmark-jblas-mmuli [n]
  (let [as (map #(DoubleMatrix/randn (pow 2 %) (pow 2 %)) (range 1 n))
        bs (map #(DoubleMatrix/randn (pow 2 %) (pow 2 %)) (range 1 n))
        cs (map #(DoubleMatrix/zeros (pow 2 %) (pow 2 %)) (range 1 n))]
    (map #((:sample-mean (criterium/quick-benchmark (.mmuli ^DoubleMatrix %2 %3 %1) {})) 0)
         as bs cs)))

(defn benchmark-vectorz-* [n]
  (let [as (map #(Matrix/createRandom (pow 2 %) (pow 2 %)) (range 1 n))
        bs (map #(Matrix/createRandom (pow 2 %) (pow 2 %)) (range 1 n))]
    (map #((:sample-mean (criterium/quick-benchmark
                          (Multiplications/multiply ^Matrix %1 ^Matrix %2) {}))
           0)
         as bs)))

(def  neanderthal-results '(1.1248269524260018E-7 1.5087644122502282E-7 3.856649117594858E-7 1.198124521836305E-6 8.037340273329063E-6 2.460520679909561E-5 1.6425813087431694E-4 0.0014111504844444445 0.009230138227272727 0.07092692575000001 0.5559446783333334 4.4479705565000005 32.518151697))

(def jblas-results '(6.871390045899843E-6 7.034027075057496E-6 7.2360799646406735E-6 8.314695169604258E-6 1.5561976039736462E-5 5.0651585861139204E-5 3.056505868686869E-4 0.0025484064875 0.018934235111111114 0.13637988016666666 1.0808136156666668 8.589283991166667 66.165910085))

(def vectorz-results '(5.737400138982502E-8 1.3233027786541422E-7 6.138861532689319E-7 3.792117722521979E-6 2.5823903242277254E-5 2.108128557894737E-4 0.0015566585 0.012041842240740742 0.09465789791666668 0.7515386003333334 6.0170086933333335 45.923629157 368.720424231))

(defn format-scaled [t]
  (let [[s u] (criterium/scale-time t)]
    (format "%.5f %s" (* s t) u)))

(defn generate-markdown-report [neanderthal-results jblas-results vectorz-results]
  (apply str
         (map #(format "| %s | %s | %s | %s | %.2f | %.2f |\n" %1 %2 %3 %4 %5 %6)
              (map #(str (long (pow 2 %)) \x (long (pow 2 %))) (range 1 (inc (count neanderthal-results))))
              (map format-scaled neanderthal-results)
              (map format-scaled jblas-results)
              (map format-scaled vectorz-results)
              (map / jblas-results neanderthal-results)
              (map / vectorz-results neanderthal-results))))
