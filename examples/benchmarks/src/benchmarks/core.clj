(ns benchmarks.core
  (:require [uncomplicate.fluokitten.core :refer [fmap!]]
            [uncomplicate.neanderthal
             [core :refer :all :exclude [entry! entry]]
             [real :refer [entry! entry]]
             [native :refer :all]
             [math :refer :all]]
            [clojure.core.matrix :as cm]
            [clojure.core.matrix.operators :as cmop]
            [criterium.core :as criterium])
  (:import [org.jblas DoubleMatrix FloatMatrix]
           [mikera.matrixx Matrix]
           [mikera.matrixx.algo Multiplications]))

(defn rnd ^double [^double x]
  (Math/random))

(defn benchmark-neanderthal-double-mm [n]
  (let [as (map #(fmap! rnd (dge (pow 2 %) (pow 2 %))) (range 1 n))
        bs (map #(fmap! rnd (dge (pow 2 %) (pow 2 %))) (range 1 n))
        cs (map #(dge (pow 2 %) (pow 2 %)) (range 1 n))]
    (map #((:sample-mean (criterium/quick-benchmark (mm! 1.0 %1 %2 0.0 %3) {})) 0) as bs cs)))

(defn benchmark-neanderthal-float-mm [n]
  (let [as (map #(fmap! rnd (sge (pow 2 %) (pow 2 %))) (range 1 n))
        bs (map #(fmap! rnd (sge (pow 2 %) (pow 2 %))) (range 1 n))
        cs (map #(sge (pow 2 %) (pow 2 %)) (range 1 n))]
    (map #((:sample-mean (criterium/quick-benchmark (mm! 1.0 %1 %2 0.0 %3) {})) 0) as bs cs)))

(defn benchmark-array-sum [n]
  (let [arrays (map #(double-array (* (pow 2 %) (* (pow 2 %)))) (range 1 n))
        sum (fn [^doubles a]
              (let [len (alength a)]
                (loop [i 0 sum 0]
                  (if (< i len)
                    (recur (inc i) (+ sum (aget ^doubles a i)))
                    sum))))]
    (map #((:sample-mean (criterium/quick-benchmark (sum %) {})) 0) arrays)))

(defn benchmark-jblas-double-mmuli [n]
  (let [as (map #(DoubleMatrix/randn (pow 2 %) (pow 2 %)) (range 1 n))
        bs (map #(DoubleMatrix/randn (pow 2 %) (pow 2 %)) (range 1 n))
        cs (map #(DoubleMatrix/zeros (pow 2 %) (pow 2 %)) (range 1 n))]
    (map #((:sample-mean
            (criterium/quick-benchmark
             (.mmuli ^DoubleMatrix %2 ^DoubleMatrix %3 ^DoubleMatrix %1) {})) 0)
         as bs cs)))

(defn benchmark-jblas-float-mmuli [n]
  (let [as (map #(FloatMatrix/randn (pow 2 %) (pow 2 %)) (range 1 n))
        bs (map #(FloatMatrix/randn (pow 2 %) (pow 2 %)) (range 1 n))
        cs (map #(FloatMatrix/zeros (pow 2 %) (pow 2 %)) (range 1 n))]
    (map #((:sample-mean
            (criterium/quick-benchmark
             (.mmuli ^FloatMatrix %2 ^FloatMatrix %3 ^FloatMatrix %1) {})) 0)
         as bs cs)))

(defn benchmark-vectorz-* [n]
  (let [as (map #(Matrix/createRandom (pow 2 %) (pow 2 %)) (range 1 n))
        bs (map #(Matrix/createRandom (pow 2 %) (pow 2 %)) (range 1 n))]
    (map #((:sample-mean (criterium/quick-benchmark
                          (Multiplications/multiply ^Matrix %1 ^Matrix %2) {}))
           0)
         as bs)))

(def neanderthal-threaded-float-results '(1.2721148453440325E-7 1.5544996654409091E-7 3.4101079478663794E-7 1.1147732724231592E-6 6.420438149955065E-6 4.7131587062333174E-5 1.9189533556832698E-4 2.5654979177268877E-4 0.0017074944833333334 0.012538112875 0.09975380433333333 0.8109630018333335 6.540657110500001))

(def  neanderthal-threaded-double-results '(1.4194661825284144E-7 1.8926280953181396E-7 4.5052947346009156E-7 1.2393542218032848E-6 8.104447407467358E-6 2.5006200233508465E-5 8.661759614301802E-5 5.635828763440861E-4 0.003755648111111112 0.025448767766666668 0.2088995005 1.6110448471666667 13.8894377865))

(def neanderthal-float-results '(1.1780910549481587E-7 1.4297028241378283E-7 3.308671157555345E-7 1.1025143436802599E-6 6.4051892350006405E-6 4.728738991331757E-5 1.91302649776928E-4 6.38760059447983E-4 0.004783855348484849 0.03800556355555556 0.287817873 2.3482393315000003 17.884675922166668))

(def neanderthal-double-results '(1.1317721266925478E-7 1.4456696261845853E-7 3.697062279474784E-7 1.1829025942212793E-6 8.235261054212755E-6 2.4232017620082068E-5 1.6426852319309602E-4 0.0012387201971544718 0.008991384277777778 0.07059805350000001 0.543674333 4.280403845166667 34.21487123483333))

(def jblas-double-results '(6.871390045899843E-6 7.034027075057496E-6 7.2360799646406735E-6 8.314695169604258E-6 1.5561976039736462E-5 5.0651585861139204E-5 3.056505868686869E-4 0.0025484064875 0.018934235111111114 0.13637988016666666 1.0808136156666668 8.589283991166667 66.165910085))

(def jblas-float-results '(3.006997907053193E-7 3.2831547984936195E-7 5.398338573175615E-7 1.5860916353606922E-6 9.038977945156052E-6 6.696848817492191E-5 2.991611432835821E-4 0.0012950075341880344 0.008533002652777778 0.06852858658333333 0.5384938281666667 4.302978593333333 273.040794861))

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
