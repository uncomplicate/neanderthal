---
title: "Benchmarks"
Author: Dragan Djuric
layout: article
---

# Matrix Multiplication: Neanderthal vs JBlas vs Vectorz (clojure.core.matrix)

The theme of this article is Neanderthal's performance. For comparisons of Matrix APIs see the tutorials.

## What is measured

Neanderthal uses [ATLAS](http://math-atlas.sourceforge.net/) with almost no overhead - it is expected that it will be as fast as your ATLAS installation (which is quite fast if set up properly).

We will concentrate on the performance of matrix-matrix multiplications, since it is the most complex operation O(n^3) (BLAS 3), and is the most telling. Neanderthal takes great care to be as light as possible, so vector (BLAS 1) and matrix-vector (BLAS 2) are also faster than in similar libraries.

We compared Neanderthal with:

* [JBlas](http://mikiobraun.github.io/jblas/) - an easy to install library that uses a staticaly compiled ATLAS BLAS implementation, and is the most popular and the most easy to install native Java matrix library. It is fast for large matrices, but slow for small matrices.
* [Vectorz](https://github.com/mikera/vectorz) - a pure Java/Clojure matrix library. Fast for small matrices, but slower for large matrices.

These two libraries are a good representation of the state of matrix computations in Java with native (JBlas) and pure Java (Vectors) libraries. They also back the most popular Clojure matrix library, [core.matrix](https://github.com/mikera/core.matrix).


## TL;DR Results

Even for very small matrices (except for matrices smaller than 5x5), Neanderthal is faster than pure Java vectorz.
For all sizes, it is faster than JBlas. For large matrices, it is 2x faster than JBlas.

## Results

The code of the benchmark is [here](https://github.com/uncomplicate/neanderthal/blob/master/examples/benchmarks/src/benchmarks/core.clj). The measurements have been done with the help of the [Criterium](https://github.com/hugoduncan/criterium) benchmarking library. On my Intel Core i7 4790k with 32GB of RAM, running Arch Linux, with Neanderthal 0.1.1 calling ATLAS 3.10.2, the results are:

| Matrix Dimensions | Neanderthal | JBlas | Vectorz | Neanderthal vs JBlas | Neanderthal vs Vectorz |
| --------------------------:| -----:| -------:| -----:| -------:| --------:|
| 2x2 | 112.48270 ns | 6.87139 µs | 57.37400 ns | 61.09 | 0.51 |
| 4x4 | 150.87644 ns | 7.03403 µs | 132.33028 ns | 46.62 | 0.88 |
| 8x8 | 385.66491 ns | 7.23608 µs | 613.88615 ns | 18.76 | 1.59 |
| 16x16 | 1.19812 µs | 8.31470 µs | 3.79212 µs | 6.94 | 3.17 |
| 32x32 | 8.03734 µs | 15.56198 µs | 25.82390 µs | 1.94 | 3.21 |
| 64x64 | 24.60521 µs | 50.65159 µs | 210.81286 µs | 2.06 | 8.57 |
| 128x128 | 164.25813 µs | 305.65059 µs | 1.55666 ms | 1.86 | 9.48 |
| 256x256 | 1.41115 ms | 2.54841 ms | 12.04184 ms | 1.81 | 8.53 |
| 512x512 | 9.23014 ms | 18.93424 ms | 94.65790 ms | 2.05 | 10.26 |
| 1024x1024 | 70.92693 ms | 136.37988 ms | 751.53860 ms | 1.92 | 10.60 |
| 2048x2048 | 555.94468 ms | 1.08081 sec | 6.01701 sec | 1.94 | 10.82 |
| 4096x4096 | 4.44797 sec | 8.58928 sec | 45.92363 sec | 1.93 | 10.32 |
| 8192x8192 | 32.51815 sec | 1.10277 min | 6.14534 min | 2.03 | 11.34 |
