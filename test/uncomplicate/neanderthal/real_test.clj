(ns uncomplicate.neanderthal.real-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.neanderthal
             [core :refer :all]
             [math :refer :all]])
  (:import [uncomplicate.neanderthal.protocols BufferAccessor]))


(defn to-buffer [^BufferAccessor accessorf s]
  (.toBuffer accessorf s))

(defmacro test-vector-constructor [accessorf vc]
  `(facts "Create a real vector."
          (vect? (~vc [1 2 3])) => true
          (~vc 1 2 3) => (~vc [1 2 3])
          (~vc 3) => (~vc [0 0 0])
          (~vc (to-buffer ~accessorf [1 2 3])) => (~vc [1 2 3])
          (~vc nil) => (throws IllegalArgumentException)
          (~vc []) => (~vc (to-buffer ~accessorf []))
          (~vc 3) => (zero (~vc [1 2 3]))))

(defmacro test-vector [vc]
  `(facts "General vector functions."
          (dim (~vc [1 2 3])) => 3
          (dim (~vc [])) => 0

          (entry (~vc [1 2 3]) 1) => 2.0
          (entry (~vc []) 0) => (throws IndexOutOfBoundsException)

          (subvector (~vc 1 2 3 4 5 6) 1 3) => (~vc 2 3 4)
          (subvector (~vc 1 2 3) 2 3) => (throws IllegalArgumentException)))

;;================ BLAS functions =========================================

(defmacro test-dot [ge vc]
  `(facts "BLAS 1 dot."
          (dot (~vc [1 2 3]) (~vc [10 30 -10])) => 40.0
          (dot (~vc [1 2 3]) nil) => (throws IllegalArgumentException)
          (dot (~vc []) (~vc [])) => 0.0
          (dot (~vc [1]) (~ge 3 3)) => (throws IllegalArgumentException)
          (dot (~vc [1 2]) (~vc [1 2 3])) => (throws IllegalArgumentException)))

(defmacro test-nrm2 [vc]
  `(facts "BLAS 1 nrm2."
          (nrm2 (~vc [1 2 3]))  => (roughly (Math/sqrt 14))
          (nrm2 (~vc [])) => 0.0))

(defmacro test-asum [vc]
  `(facts "BLAS 1 asum."
          (asum (~vc [1 2 -3]))  => 6.0
          (asum (~vc [])) => 0.0))

(defmacro test-sum [vc]
  `(facts "BLAS Plus 1 sum."
          (sum (~vc [1 2 -5]))  => -2.0
          (sum (~vc [])) => 0.0))

(defmacro test-iamax [vc]
  `(facts "BLAS 1 iamax"
          (iamax (~vc [1 2 7 7 6 2 -10 10])) => 6
          (iamax (~vc [])) => 0
          (iamax (~vc [4 6 7 3])) => 2))

(defmacro test-rot [vc]
  `(facts "BLAS 1 rot!"
n          (let [x# (~vc [1 2 3 4 5])
                y# (~vc [-1 -2 -3 -4 -5])]
            (do (rot! x# y# 1 0) [x# y#]) => [(~vc 1 2 3 4 5) (~vc -1 -2 -3 -4 -5)]

            (do (rot! x# y# 0.5 (/ (sqrt 3) 2.0)) [x# y#]
                (+ (double (nrm2 (axpy -1 x# (~vc -0.3660254037844386 -0.7320508075688772
                                                  -1.098076211353316 -1.4641016151377544
                                                  -1.8301270189221928))))
                   (double (nrm2 (axpy -1 y# (~vc -1.3660254037844386 -2.732050807568877
                                                  -4.098076211353316 -5.464101615137754
                                                  -6.830127018922193))))))
            => (roughly 0 1e-6)

            (rot! x# y# (/ (sqrt 3) 2.0)  0.5) => x#
            (rot! x# (~vc 1) 0.5 0.5) => (throws IllegalArgumentException)
            (rot! x# y# 300 0.3) => (throws IllegalArgumentException)
            (rot! x# y# 0.5 0.5) => (throws IllegalArgumentException))))

(defmacro test-rotg [vc]
  `(facts "BLAS 1 rotg!"
          (let [x# (~vc 1 2 3 4)]
            (rotg! x#) => x#)
          (rotg! (~vc 0 0 0 0)) => (~vc 0 0 1 0)
          (rotg! (~vc 0 2 0 0)) => (~vc 2 1 0 1)
          (nrm2 (axpy -1 (rotg! (~vc 6 -8 0 0)) (~vc -10 -5/3 -3/5 4/5)))
          => (roughly 0 1e-6)))

(defmacro test-rotm [vc]
  `(facts "BLAS 1 rotm!"
          "TODO: Leave this for later since I am lacking good examples right now."))

(defmacro test-rotmg [vc]
  `(facts "BLAS 1 rotmg!"
          "TODO: Leave this for later since I am lacking good examples right now."))

(defmacro test-swap [ge vc]
  `(facts "BLAS 1 swap! vectors"
          (let [x# (~vc [1 2 3])]
            (swp! x# (~vc 3)) => x#
            (swp! x# (~vc [10 20 30])) => (~vc [10 20 30])
            (swp! x# nil) => (throws IllegalArgumentException)
            (identical? (swp! x# (~vc [10 20 30])) x#) => true
            (swp! x# (~vc [10 20])) => (throws IllegalArgumentException)))

  `(facts
    "BLAS 1 swap! matrices"
    (let [a# (~ge 2 3 [1 2 3 4 5 6])]
      (swp! a# (~ge 2 3)) => a#
      (swp! a# (~ge 2 3 [10 20 30 40 50 60])) => (~ge 2 3 [10 20 30 40 50 60])
      (swp! a# nil) => (throws IllegalArgumentException)
      (identical? (swp! a# (~ge 2 3 [10 20 30 40 50 60])) a#) => true
      (swp! a# (~ge 1 2 [10 20])) => (throws IllegalArgumentException))))

(defmacro test-scal [vc]
  `(facts "BLAS 1 scal!"
          (let [x# (~vc [1 2 3])]()
               (identical? (scal! 3 x#) x#) => true)
          (scal! 3 (~vc [1 -2 3])) => (~vc [3 -6 9])
          (scal! 3 (~vc [])) => (~vc [])))

(defmacro test-copy [vc]
  `(facts "BLAS 1 copy!"
          (let [y# (~vc 3)]
            (copy! (~vc [1 2 3]) y#) => y#
            (copy (~vc [1 2 3]) => y#))

          (copy! (~vc [10 20 30]) (~vc [1 2 3])) => (~vc [10 20 30])
          (copy! (~vc [1 2 3]) nil) => (throws IllegalArgumentException)
          (copy! (~vc [10 20 30]) (~vc [1])) => (throws IllegalArgumentException))

  `(facts "BLAS 1 copy"
          (copy (~vc 1 2)) => (~vc 1 2)
          (let [x# (~vc 1 2)] (identical? (copy x#) x#) => false)))

(defmacro test-axpy [vc]
  `(facts "BLAS 1 axpy!"
          (let [y# (~vc [1 2 3])]
            (axpy! 2.0 (~vc [10 20 30]) y#) => (~vc [21 42 63])
            (identical? (axpy! 3.0 (~vc [1 2 3]) y#) y#) => true
            (axpy! 2.0 (~vc [1 2]) y#) => (throws IllegalArgumentException)
            (axpy! (~vc [10 20 30]) (~vc [1 2 3])) => (~vc [11 22 33]))

          (axpy! 2 (~vc 1 2 3) (~vc 1 2 3) (~vc 2 3 4) 4.0 (~vc 5 6 7))
          => (~vc 25 33 41)
          (axpy! 2 (~vc 1 2 3) (~vc 1 2 3) (~vc 2 3 4) 4.0)
          => (throws IllegalArgumentException)
          (axpy! (~vc 1 2 3) (~vc 1 2 3) (~vc 1 2 3) (~vc 1 2 3) (~vc 1 2 3))
          => (axpy! 5 (~vc 1 2 3) (~vc 3))
          (axpy! 4 "af" (~vc 1 2 3) 3 "b" "c")
          => (throws IllegalArgumentException)
          (let [y# (~vc [1 2 3])]
            (axpy! 2 (~vc 1 2 3) y# (~vc 2 3 4) 4.0 (~vc 5 6 7)) => y#))

  `(facts "BLAS1 axpy"
          (let [y# (~vc [1 2 3])]
            (axpy 2.0 (~vc [10 20 30])) => (throws IllegalArgumentException)
            (identical? (axpy 3.0 (~vc [1 2 3]) y#) y#) => false
            (axpy (~vc 1 2 3) (~vc 2 3 4) (~vc 3 4 5) (~vc 3)) => (~vc 6 9 12)
            (axpy 2 (~vc 1 2 3) (~vc 2 3 4)) => (~vc 4 7 10))))

;; ================= Real Matrix functions =====================================

(defmacro test-matrix-constructor [accessorf ge]
  `(facts "Create a matrix."
          (let [a# (~ge 2 3 [1 2 3 4 5 6])]
            (~ge 2 3 [1 2 3 4 5 6]) => a#
            (~ge 2 3 (to-buffer ~accessorf [1 2 3 4 5 6])) => a#
            (~ge 2 3 nil) => (throws IllegalArgumentException)
            (~ge 0 0 []) => (~ge 0 0 (to-buffer ~accessorf [])))))

(defmacro test-matrix [ge vc]
  `(facts "General Matrix methods."
          (ncols (~ge 2 3 [1 2 3 4 5 6])) => 3
          (ncols (~ge 0 0 [])) => 0

          (mrows (~ge 2 3)) => 2
          (mrows (~ge 0 0)) => 0

          (entry (~ge 2 3 [1 2 3 4 5 6]) 0 1) => 3.0
          (entry (~ge 0 0) 0 0) => (throws IndexOutOfBoundsException)
          (entry (~ge 2 3 [1 2 3 4 5 6]) -1 2) => (throws IndexOutOfBoundsException)
          (entry (~ge 2 3 [1 2 3 4 5 6]) 2 1) => (throws IndexOutOfBoundsException)

          (row (~ge 2 3 [1 2 3 4 5 6]) 1) => (~vc 2 4 6)
          (row (~ge 2 3 [1 2 3 4 5 6]) 4) => (throws IndexOutOfBoundsException)

          (col (~ge 2 3 [1 2 3 4 5 6]) 1) => (~vc 3 4)
          (col (~ge 2 3 [1 2 3 4 5 6]) 4) => (throws IndexOutOfBoundsException)

          (submatrix (~ge 3 4 (range 12)) 1 2 2 1) => (~ge 2 1 [7 8])
          (submatrix (~ge 3 4 (range 12)) 2 3) => (~ge 2 3 [0 1 3 4 6 7])
          (submatrix (~ge 3 4 (range 12)) 3 4) => (~ge 3 4 (range 12))
          (submatrix (~ge 3 4 (range 12)) 1 1 3 3)
          => (throws IndexOutOfBoundsException)))

;; ====================== BLAS 2 ===============================

(defmacro test-mv [ge vc]
  `(facts "BLAS 2 mv!"
          (mv! 2.0 (~ge 3 2 [1 2 3 4 5 6]) (~vc 1 2 3) 3 (~vc [1 2 3 4]))
          => (throws IllegalArgumentException)

          (let [y# (~vc [1 2 3])]
            (mv! 2 (~ge 3 2 [1 2 3 4 5 6]) (~vc 1 2) 3 y#)
            => y#)

          (mv! (~ge 2 3 [1 10 2 20 3 30]) (~vc 7 0 4) (~vc 0 0)) => (~vc 19 190)

          (mv! 2.0 (~ge 2 3 [1 2 3 4 5 6]) (~vc 1 2 3) 3.0 (~vc 1 2))
          => (~vc 47 62)

          (mv! 2.0 (~ge 2 3 [1 2 3 4 5 6]) (~vc 1 2 3) 0.0 (~vc 0 0))
          => (mv 2.0 (~ge 2 3 [1 2 3 4 5 6]) (~vc 1 2 3))

          (mv! (trans (~ge 3 2 [1 3 5 2 4 6])) (~vc 1 2 3) (~vc 0 0))
          => (mv! (~ge 2 3 [1 2 3 4 5 6]) (~vc 1 2 3) (~vc 0 0))))

(defmacro test-rank [ge vc]
  `(facts "BLAS 2 rank!"
          (let [a# (~ge 2 3 [1 2 3 4 5 6])]
            (rank! 2.0 (~vc 1 2) (~vc 1 2 3) a#) => a#)
          (rank! (~vc 3 2 1 4) (~vc 1 2 3) (~ge 4 3 [1 2 3 4 2 2 2 2 3 4 2 1]))
          => (~ge 4 3 [4 4 4 8 8 6 4 10 12 10 5 13])

          (rank! (~vc 1 2) (~vc 1 2 3) (~ge 2 2 [1 2 3 5]))
          => (throws IllegalArgumentException)))

;; ====================== BLAS 3 ================================

(defmacro test-mm [ge]
  `(facts "BLAS 3 mm!"
          (mm! 2.0 (~ge 3 2 [1 2 3 4 5 6]) (~ge 3 2 [1 2 3 4 5 6])
               3.0 (~ge 3 2 [1 2 3 4 5 6]))
          => (throws IllegalArgumentException)

          (mm! 2.0 (~ge 3 2 [1 2 3 4 5 6]) (~ge 2 3 [1 2 3 4 5 6])
               3.0 (~ge 3 2 [1 2 3 4 5 6]))
          => (throws IllegalArgumentException)

          (let [c (~ge 2 2 [1 2 3 4])]
            (mm! 2.0 (~ge 2 3 [1 2 3 4 5 6]) (~ge 3 2 [1 2 3 4 5 6]) 3.0 c)
            => c)

          (mm! 2.0 (~ge 2 3 [1 2 3 4 5 6]) (~ge 3 2 [1 2 3 4 5 6])
               3.0 (~ge 2 2 [1 2 3 4]))
          => (~ge 2 2 [47 62 107 140])

          (mm! 2.0 (trans (~ge 3 2 [1 3 5 2 4 6])) (trans (~ge 2 3 [1 4 2 5 3 6]))
               3.0  (~ge 2 2 [1 2 3 4]))
          => (~ge 2 2 [47 62 107 140]))

  `(facts "mm"
          (mm (~ge 3 2 (range 6)) (~ge 2 3 (range 6)))
          => (~ge 3 3 [3 4 5 9 14 19 15 24 33])))

(defmacro test-all [accessorf ge vc]
  `(do (test-vector-constructor ~accessorf ~vc)
       (test-vector ~vc)
       (test-dot ~ge ~vc)
       (test-nrm2 ~vc)
       (test-asum ~vc)
       (test-sum ~vc)
       (test-iamax ~vc)
       (test-rot ~vc)
       (test-rotg ~vc)
       (test-rotm ~vc)
       (test-rotmg ~vc)
       (test-swap ~ge ~vc)
       (test-scal ~vc)
       (test-copy ~vc)
       (test-axpy ~vc)
       (test-matrix-constructor ~accessorf ~ge)
       (test-matrix ~ge ~vc)
       (test-mv ~ge ~vc)
       (test-rank ~ge ~vc)
       (test-mm ~ge)))
