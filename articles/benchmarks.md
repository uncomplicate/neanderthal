---
title: "Benchmarks"
Author: Dragan Djuric
layout: article
---

# Matrix Multiplication: Neanderthal vs JBlas vs Vectorz (clojure.core.matrix)

## What is measured

Neanderthal uses ATLAS with almost no overhead - it is expected that it will be as fast as your ATLAS installation (which is quite fast if set up properly).

We will concentrate on the performance of matrix-matrix multiplication, since it is the most complex operation O(n^3) (BLAS 3), and is the most telling. Neanderthal takes great care to be as light as possible, so vector (BLAS 1) and matrix-vector (BLAS 2) are also faster than in similar libraries.

We compared Neanderthal with:

* JBlas - an easy to install library that uses native BLAS implementation, and is the most popular and easy to install native Java matrix library. It is fast for large matrices, but slow for small matrices. It is used by `clojure.core.matrix library;
* Vectorz - a pure java/clojure matrix library. Fast for small matrices, but slow for large matrices. Also used by `clojure.core.matrix;
* Java vector summation - for reference, we also measure the time needed to sum a primitive array that has the length equal to the dimension of the matrix.


## TL;DR Results

Even for very small matrices (except for matrices smaller than 5x5), Neanderthal is faster than pure Java vectorz.
For all sizes, it is faster than JBlas.
For large matrices, it is 2x faster than JBlas.

## Results

The code of the benchmark is [here]. The measurements have been done with the help of criterium benchmarking library. On my Intel Core i7 4790k, the results are as follows, measured in seconds:

| Neanderthal __________________ | JBLAS | Vectorz |
| -------------------------- | ----- | ------- | -----
| 1.1248269524260018E-7 | 6.871390045899843E-6 | 5.737400138982502E-8 |
| 1.5087644122502282E-7 | 7.034027075057496E-6 | 1.3233027786541422E-7 |
| 3.856649117594858E-7 | 7.2360799646406735E-6 | 6.138861532689319E-7 |
| 1.198124521836305E-6 | 8.314695169604258E-6 | 3.792117722521979E-6 |
| 8.037340273329063E-6 | 1.5561976039736462E-5 | 2.5823903242277254E-5 |
| 2.460520679909561E-5 | 5.0651585861139204E-5 | 2.108128557894737E-4 |
| 1.6425813087431694E-4 | 3.056505868686869E-4 | 0.0015566585 |
| 0.0014111504844444445 | 0.0025484064875 | 0.012041842240740742 |
| 0.009230138227272727 | 0.018934235111111114 | 0.09465789791666668 |
| 0.07092692575000001 | 0.13637988016666666 | 0.7515386003333334 |
| 0.5559446783333334 | 1.0808136156666668 | 6.0170086933333335 |
| 4.4479705565000005 | 8.589283991166667 | 45.923629157 |
| 32.518151697 | 66.165910085 | 368.720424231 |
