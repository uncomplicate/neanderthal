---
title: "Benchmarks"
Author: Dragan Djuric
layout: article
---

## Matrix Multiplication: Neanderthal vs JBlas vs Vectorz (clojure.core.matrix)

The theme of this article is Neanderthal's performance. For comparisons of Matrix APIs, see the tutorials.

### TL;DR Results

Even for very small matrices (except for matrices smaller than 5x5), Neanderthal is faster than pure Java vectorz.
For all sizes, it is faster than jBLAS. For large matrices, it is **2x faster than jBLAS**. **Neanderthal uses single-threaded ATLAS by default**, it can be even faster with parallel ATLAS, but for my use cases, I prefer to leave parallelism in Clojure parts of the app.

### What is Being Measured

Neanderthal uses [ATLAS](http://math-atlas.sourceforge.net/) with almost no overhead - it is expected that it will be as fast as your ATLAS installation (which is quite fast if set up properly).

I will concentrate on the performance of matrix-matrix multiplications, since it is the most complex operation O(n^3) (BLAS 3), and is the most telling. Neanderthal takes great care to be as light as possible, and does not use data copying, so even linear operations, vector (BLAS 1) and quadratic matrix-vector (BLAS 2) are faster than in pure Java.

I compared Neanderthal with:

* [jBLAS](http://mikiobraun.github.io/jblas/) - an easy to install library that uses a staticaly compiled ATLAS BLAS implementation, and is the most popular and the most easy to install native Java matrix library. It is fast for large matrices, but slow for small matrices.
* [Vectorz](https://github.com/mikera/vectorz) - a pure Java/Clojure matrix library. Fast for small matrices, but slower for large matrices.

These two libraries are a good representation of the state of matrix computations in Java with native (jBLAS) and pure Java (Vectors) libraries. They also back the most popular Clojure matrix library, [core.matrix](https://github.com/mikera/core.matrix).


### Results

The code of the benchmark is [here](https://github.com/uncomplicate/neanderthal/blob/master/examples/benchmarks/src/benchmarks/core.clj). The measurements have been done with the help of the [Criterium](https://github.com/hugoduncan/criterium) benchmarking library. On my Intel Core i7 4790k with 32GB of RAM, running Arch Linux, with Neanderthal 0.1.1 calling ATLAS 3.10.2, the results are:

| Matrix Dimensions | Neanderthal | jBLAS | Vectorz | Neanderthal vs jBLAS | Neanderthal vs Vectorz |
| --------------------------:| -----:| -------:| -----:| -------:| --------:|
| 2x2 | 112 ns | 6.9 µs | 57 ns | 61.09 | 0.51 |
| 4x4 | 151 ns | 7.0 µs | 132 ns | 46.62 | 0.88 |
| 8x8 | 385 ns | 7.2 µs | 614 ns | 18.76 | 1.59 |
| 16x16 | 1.2 µs | 8.31 µs | 3.79 µs | 6.94 | 3.17 |
| 32x32 | 8.0 µs | 15.6 µs | 25.8 µs | 1.94 | 3.21 |
| 64x64 | 24.6 µs | 50.7 µs | 211 µs | 2.06 | 8.57 |
| 128x128 | 164 µs | 306 µs | 1.56 ms | 1.86 | 9.48 |
| 256x256 | 1.4 ms | 2.55 ms | 12.0 ms | 1.81 | 8.53 |
| 512x512 | 9.2 ms | 18.9 ms | 94.7 ms | 2.05 | 10.26 |
| 1024x1024 | 71 ms | 136 ms | 752 ms | 1.92 | 10.60 |
| 2048x2048 | 556 ms | 1.08 sec | 6.02 sec | 1.94 | 10.82 |
| 4096x4096 | 4.45 sec | 8.59 sec | 45.9 sec | 1.93 | 10.32 |
| 8192x8192 | 32.5 sec | 1.10 min | 6.14 min | 2.03 | 11.34 |
