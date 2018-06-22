(ns benchmarks.core
  (:require [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer :all :exclude [entry! entry]]
             [real :refer [entry! entry]]
             [native :refer :all]
             [math :refer :all]]
            [clojure.core.matrix :as cm]
            [clojure.core.matrix.operators :as cmop]
            [clatrix.core :as clatrix]
            [criterium.core :as criterium]
            [clojure.pprint :refer [cl-format]])
  (:import [org.jblas DoubleMatrix FloatMatrix]
          [mikera.matrixx Matrix]
           [mikera.matrixx.algo Multiplications]))

(defn rnd ^double [^double x]
  (Math/random))

(defn benchmark-neanderthal-double-mm [f n0 n]
  (let [as (map #(fmap! rnd (dge (f (pow 2 %)) (f (pow 2 %)))) (range n0 n))
        bs (map #(fmap! rnd (dge (f (pow 2 %)) (f (pow 2 %)))) (range n0 n))
        cs (map #(dge (f (pow 2 %)) (f (pow 2 %))) (range n0 n))]
    (map #((:sample-mean (criterium/quick-benchmark (mm! 1.0 %1 %2 0.0 %3) {})) 0) as bs cs)))

(defn benchmark-neanderthal-float-mm [f n0 n]
  (let [as (map #(fmap! rnd (fge (f (pow 2 %)) (f (pow 2 %)))) (range n0 n))
        bs (map #(fmap! rnd (fge (f (pow 2 %)) (f (pow 2 %)))) (range n0 n))
        cs (map #(fge (f (pow 2 %)) (f (pow 2 %))) (range n0 n))]
    (map #((:sample-mean (criterium/quick-benchmark (mm! 1.0 %1 %2 0.0 %3) {})) 0) as bs cs)))

(defn benchmark-array-sum [f n0 n]
  (let [arrays (map #(double-array (* (f (pow 2 %)) (* (f (pow 2 %))))) (range n0 n))
        sum (fn [^doubles a]
              (let [len (alength a)]
                (loop [i 0 sum 0]
                  (if (< i len)
                    (recur (inc i) (+ sum (aget ^doubles a i)))
                    sum))))]
    (map #((:sample-mean (criterium/quick-benchmark (sum %) {})) 0) arrays)))

(defn benchmark-jblas-double-mmuli [f n0 n]
  (let [as (map #(DoubleMatrix/randn (f (pow 2 %)) (f (pow 2 %))) (range n0 n))
        bs (map #(DoubleMatrix/randn (f (pow 2 %)) (f (pow 2 %))) (range n0 n))
        cs (map #(DoubleMatrix/zeros (f (pow 2 %)) (f (pow 2 %))) (range n0 n))]
    (map #((:sample-mean
            (criterium/quick-benchmark
             (.mmuli ^DoubleMatrix %2 ^DoubleMatrix %3 ^DoubleMatrix %1) {})) 0)
         as bs cs)))

(defn benchmark-jblas-float-mmuli [f n0 n]
  (let [as (map #(FloatMatrix/randn (f (pow 2 %)) (f (pow 2 %))) (range n0 n))
        bs (map #(FloatMatrix/randn (f (pow 2 %)) (f (pow 2 %))) (range n0 n))
        cs (map #(FloatMatrix/zeros (f (pow 2 %)) (f (pow 2 %))) (range n0 n))]
    (map #((:sample-mean
            (criterium/quick-benchmark
             (.mmuli ^FloatMatrix %2 ^FloatMatrix %3 ^FloatMatrix %1) {})) 0)
         as bs cs)))

(defn benchmark-vectorz-* [f n0 n]
  (let [as (map #(Matrix/createRandom (f (pow 2 %)) (f (pow 2 %))) (range n0 n))
        bs (map #(Matrix/createRandom (f (pow 2 %)) (f (pow 2 %))) (range n0 n))]
    (map #((:sample-mean (criterium/quick-benchmark
                          (Multiplications/multiply ^Matrix %1 ^Matrix %2) {}))
           0)
         as bs)))

(defn benchmark-clatrix-* [f n0 n]
  (let [as (map #(clatrix/rand (f (pow 2 %)) (f (pow 2 %))) (range n0 n))
        bs (map #(clatrix/rand (f (pow 2 %)) (f (pow 2 %))) (range n0 n))]
    (map #((:sample-mean (criterium/quick-benchmark (cm/mmul %1 %2) {}))
           0)
         as bs)))


(def neanderthal-float-results '(2.3236308878251278E-7 2.3772209383145782E-7 2.5322398444460806E-7 3.7229697243366456E-7 9.03142077526016E-7 2.7954640232941506E-6 1.6297250691844376E-5 1.2624948442367602E-4 0.0010746654543650795 0.007934090911764705 0.05747208241666667 0.4701220286666667 3.759669557))

(def neanderthal-double-results '(2.2865523231789207E-7 2.2928626754198731E-7 2.638415951500431E-7 4.288042939570632E-7 1.5154079119367047E-6 6.389600786568445E-6 4.368230687551332E-5 3.3197504918032787E-4 0.003553800108333334 0.020049612476190477 0.1527479125 0.9993730395000001 8.836732611))

(def jblas-double-results '(3.6046276400467775E-7 4.339420231836879E-7 4.7594740812022817E-7 7.616580659679314E-7 5.079253627390074E-6 1.4446252529286477E-5 7.095677674684995E-5 4.970345309050773E-4 0.004018943086666667 0.0260148844 0.16591023533333335 1.491862468 11.727497799666667))

(def jblas-float-results '(3.6200474525957515E-7 3.699931665115261E-7 4.7657205100522573E-7 5.984288469555971E-7 1.3684475723534748E-6 7.52324758971277E-6 3.147610978638847E-5 1.911487552552553E-4 0.0012515618364197533 0.010626416347222225 0.10494894975 0.5684600615000001 4.852702502666667))

(def vectorz-results '(6.135846356300698E-8 1.293400802613693E-7 5.680150690716849E-7 3.450187949095811E-6 2.344169735023042E-5 2.186445752071383E-4 0.001545798075757576 0.012276579296296298 0.09694332241666667 0.7784629151666667 6.216332004500001 50.06204500516667 400.618242))

(def clatrix-results '(7.62321650739902E-7 8.15511833847966E-7 8.803057710988614E-7 1.4350191870086885E-6 5.453781994780595E-6 1.7962920889737362E-5 7.904317079990275E-5 4.7761557876230666E-4 0.0044436613680555554 0.039357244444444446 0.15437532366666668 1.0640010516666667 9.239831300166667))

(defn format-scaled [t]
  (let [[s u] (criterium/scale-time t)]
    (cl-format nil "~6,2F ~a" (* s t) u)))

(defn generate-markdown-report [f neanderthal-results clatrix-results vectorz-results]
  (apply str
         (map #(format "| %s | %s | %s | %s | %.2f | %.2f |\n" %1 %2 %3 %4 %5 %6)
              (map #(str (long (f (pow 2 %))) \x (long (f (pow 2 %)))) (range 1 (inc (count neanderthal-results))))
              (map format-scaled neanderthal-results)
              (map format-scaled clatrix-results)
              (map format-scaled vectorz-results)
              (map / clatrix-results neanderthal-results)
              (map / vectorz-results neanderthal-results))))
