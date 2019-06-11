;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(ns uncomplicate.neanderthal.random-test
  (:require [midje.sweet :refer [facts throws => roughly]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal
             [core :refer [col row sum vctr ge dim submatrix raw transfer]]
             [random :refer :all]])
  (:import clojure.lang.ExceptionInfo))

(defn test-vctr-rand-uniform [factory]
  (facts "Test vector rand-uniform."
         (let [m 9999
               n 99]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)
                          x0 (vctr factory 0)]
             (/ (double (sum (rand-uniform! seed 0 1 (col a 0)))) m)
             => (roughly 0.5 0.01)
             (/ (double (sum (rand-uniform! seed 0 100 (col a 0)))) m)
             => (roughly 50 1)
             (/ (double (sum (rand-uniform! seed -100 10 (col a 0)))) m)
             => (roughly -45 1)
             (rand-normal! seed 0 1 x0) => x0))))

(defn test-vctr-rand-normal [factory]
  (facts "Test vector rand-normal."
         (let [m 9999
               n 99]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)
                          x0 (vctr factory 0)]
             (/ (double (sum (rand-normal! seed 0 1 (col a 0)))) m)
             => (roughly 0 0.03)
             (/ (double (sum (rand-normal! seed 0 100 (col a 0)))) m)
             => (roughly 0 1)
             (/ (double (sum (rand-normal! seed -100 1 (col a 0)))) m)
             => (roughly -100 0.03)
             (rand-normal! seed 0 1 x0) => x0))))

(defn test-vctr-rand-host [factory]
  (facts "Test vector rand methods on the host."
         (let [m 99
               n 77]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)]
             (rand-uniform! seed 0 1 (row a 1)) => (throws ExceptionInfo)
             (rand-normal! seed 0 1 (row a 1)) => (throws ExceptionInfo)))))

(defn test-vctr-rand-uniform-device [factory]
  (facts "Test vector rand-uniform on a device."
         (let [m 99
               n 77]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)]
             (/ (double (sum (rand-uniform! seed -100 100 (row a 1)))) n)
             => (roughly 0 1)
             (/ (double (sum (rand-uniform! seed 0 1 (row a 1)))) n)
             => (roughly 0.5 0.1)))))

(defn test-vctr-rand-normal-device [factory]
  (facts "Test vector rand-normal on a device."
         (let [m 99
               n 77]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)]
             (/ (double (sum (rand-normal! seed -100 2 (row a 1)))) n)
             => (roughly -100 0.3)
             (/ (double (sum (rand-normal! seed 0 1 (row a 1)))) n)
             => (roughly 0 0.3)))))

(defn test-ge-rand-uniform [factory]
  (facts "Test GE matrix rand-uniform."
         (let [m 9999
               n 99]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)
                          b (raw (submatrix a 0 5 9999 77))]
             (/ (double (sum (rand-uniform! seed 0 1 a))) (dim a))
             => (roughly 0.5 0.01)
             (/ (double (sum (rand-uniform! seed 0 100 a))) (dim a))
             => (roughly 50 1)
             (/ (double (sum (transfer (rand-uniform! seed -100 10 b)))) (dim b))
             => (roughly -45 1)))))

(defn test-ge-rand-normal [factory]
  (facts "Test GE matrix rand-normal."
         (let [m 9999
               n 99]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)
                          b (raw (submatrix a 0 5 9999 77))]
             (/ (double (sum (rand-normal! seed 0 1 a))) (dim a))
             => (roughly 0 0.03)
             (/ (double (sum (rand-normal! seed 0 100 a))) (dim a))
             => (roughly 0 1)
             (/ (double (sum (transfer (rand-normal! seed -100 10 b)))) (dim b))
             => (roughly -100 0.03)))))

(defn test-ge-rand-host [factory]
  (facts "Test GE matrix rand-uniform on the host."
         (let [m 9999
               n 99]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)
                          b (raw (submatrix a 3 5 89 77))]
             (rand-uniform! seed -100 10 b) => (throws ExceptionInfo)
             (rand-normal! seed -100 10 b) => (throws ExceptionInfo)))))

(defn test-ge-rand-uniform-device [factory]
  (facts "Test GE matrix rand-uniform on a device."
         (let [m 9999
               n 99]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)
                          b (raw (submatrix a 3 5 89 77))]
             (/ (double (sum (transfer (rand-uniform! seed -100 100 b)))) (dim b))
             => (roughly 0 1)
             (/ (double (sum (transfer (rand-uniform! seed 0 1 b)))) (dim b))
             => (roughly 0.5 0.1)))))

(defn test-ge-rand-normal-device [factory]
  (facts "Test GE matrix rand-normal on a device."
         (let [m 9999
               n 99]
           (with-release [seed (rng-state factory 42)
                          a (ge factory m n)
                          b (raw (submatrix a 3 5 89 77))]
             (/ (double (sum (transfer (rand-normal! seed -100 2 b)))) (dim b))
             => (roughly -100 0.3)
             (/ (double (sum (transfer (rand-normal! seed 0 1 b)))) (dim b))
             => (roughly 0 0.3)))))

(defn test-all [factory]
  (test-vctr-rand-uniform factory)
  (test-vctr-rand-normal factory)
  (test-ge-rand-uniform factory)
  (test-ge-rand-normal factory))

(defn test-all-device [factory]
  (test-vctr-rand-uniform-device factory)
  (test-vctr-rand-normal-device factory)
  (test-ge-rand-uniform-device factory)
  (test-ge-rand-normal-device factory))

(defn test-all-host [factory]
  (test-vctr-rand-host factory)
  (test-ge-rand-host factory))
