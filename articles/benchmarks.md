---
title: "Benchmarks"
Author: Dragan Djuric
layout: article
---

## Matrix Multiplication: Neanderthal vs JBlas vs Vectorz (clojure.core.matrix)

The theme of this article is Neanderthal's performance on the CPU. For comparisons of Matrix APIs, or GPU performance, see the tutorials (more than 1000x speedup for some operations).

### TL;DR Results

* **Even for very small matrices** (except for matrices smaller than 5x5), Neanderthal is faster than core.matrix's flagship implementation (pure Java library Vectorz). For large matrices, Neanderthal is **much** faster than the flagship core.matrix implementation Vectorz: **60x faster** (floats), 30x faster (doubles), 20x faster (single-threaded floats), and 10x faster (single-threaded doubles).

* For large matrices, it is **5x faster than jBLAS** (multi-threaded) and **2x faster** (single-threaded) . For all sizes, it is much faster than jBLAS.

### What is Being Measured

Neanderthal uses [ATLAS](http://math-atlas.sourceforge.net/) with almost no overhead - it is expected that it will be as fast as your ATLAS installation (which is quite fast if set up properly).

I will concentrate on the performance of matrix-matrix multiplications, since it is the most complex operation O(n^3) (BLAS 3), and is the most telling. Neanderthal takes great care to be as light as possible, and does not use data copying, so even linear operations, vector (BLAS 1) and quadratic matrix-vector (BLAS 2) are faster than in pure Java.

I compared Neanderthal with:

* [jBLAS](http://mikiobraun.github.io/jblas/) - an easy to install library that uses a staticaly compiled ATLAS BLAS implementation, and is the most popular and the easiest to install native Java matrix library. It is fast for large matrices, but slow for small matrices.
* [Vectorz](https://github.com/mikera/vectorz) - a pure Java/Clojure matrix library. Fast for small matrices, but slower for large matrices.

These two libraries are a good representation of the state of matrix computations in Java with native (jBLAS) and pure Java (Vectors) libraries. They also back the most popular Clojure matrix library, [core.matrix](https://github.com/mikera/core.matrix).

### Results

The code of the benchmark is [here](https://github.com/uncomplicate/neanderthal/blob/master/examples/benchmarks/src/benchmarks/core.clj). The measurements have been done with the help of the [Criterium](https://github.com/hugoduncan/criterium) benchmarking library. On my Intel Core i7 4790k with 32GB of RAM, running Arch Linux, with Neanderthal 0.5.0 calling ATLAS 3.10.2, the results are:

#### Threaded ATLAS, single floating-point precision:

| Matrix Dimensions | Neanderthal | jBLAS | Vectorz | Neanderthal vs jBLAS | Neanderthal vs Vectorz |
| -----------------:| -----------:| -----:| -------:| --------------------:| ----------------------:|
| 2x2 | 127.21148 ns | 300.69979 ns | 57.37400 ns | 2.36 | 0.45 |
| 4x4 | 155.44997 ns | 328.31548 ns | 132.33028 ns | 2.11 | 0.85 |
| 8x8 | 341.01079 ns | 539.83386 ns | 613.88615 ns | 1.58 | 1.80 |
| 16x16 | 1.11477 µs | 1.58609 µs | 3.79212 µs | 1.42 | 3.40 |
| 32x32 | 6.42044 µs | 9.03898 µs | 25.82390 µs | 1.41 | 4.02 |
| 64x64 | 47.13159 µs | 66.96849 µs | 210.81286 µs | 1.42 | 4.47 |
| 128x128 | 191.89534 µs | 299.16114 µs | 1.55666 ms | 1.56 | 8.11 |
| 256x256 | 256.54979 µs | 1.29501 ms | 12.04184 ms | 5.05 | 46.94 |
| 512x512 | 1.70749 ms | 8.53300 ms | 94.65790 ms | 5.00 | 55.44 |
| 1024x1024 | 12.53811 ms | 68.52859 ms | 751.53860 ms | 5.47 | 59.94 |
| 2048x2048 | 99.75380 ms | 538.49383 ms | 6.01701 sec | 5.40 | 60.32 |
| 4096x4096 | 810.96300 ms | 4.30298 sec | 45.92363 sec | 5.31 | 56.63 |
| 8192x8192 | 6.54066 sec | 4.55068 min | 6.14534 min | 41.75 | 56.37 |

#### Threaded ATLAS, double floating-point precision:

| Matrix Dimensions | Neanderthal | jBLAS | Vectorz | Neanderthal vs jBLAS | Neanderthal vs Vectorz |
| -----------------:| -----------:| -----:| -------:| --------------------:| ----------------------:|
| 2x2 | 141.94 ns | 6.87 µs | 57.37 ns | 48.41 | 0.40 |
| 4x4 | 189.261 ns | 7.03 µs | 132.33 ns | 37.17 | 0.70 |
| 8x8 | 450.52 ns | 7.23 µs | 613.88 ns | 16.06 | 1.36 |
| 16x16 | 1.23 µs | 8.31 µs | 3.79 µs | 6.71 | 3.06 |
| 32x32 | 8.10 µs | 15.56 µs | 25.82 µs | 1.92 | 3.19 |
| 64x64 | 25.00 µs | 50.65 µs | 210.81 µs | 2.03 | 8.43 |
| 128x128 | 86.61 µs | 305.65 µs | 1.55 ms | 3.53 | 17.97 |
| 256x256 | 563.58 µs | 2.54 ms | 12.04 ms | 4.52 | 21.37 |
| 512x512 | 3.75 ms | 18.93 ms | 94.65 ms | 5.04 | 25.20 |
| 1024x1024 | 25.44 ms | 136.37 ms | 751.53 ms | 5.36 | 29.53 |
| 2048x2048 | 208.90 ms | 1.08 sec | 6.01 sec | 5.17 | 28.80 |
| 4096x4096 | 1.61 sec | 8.58 sec | 45.92 sec | 5.33 | 28.51 |
| 8192x8192 | 13.89 sec | 1.10 min | 6.14 min | 4.76 | 26.55 |

#### With serial ATLAS, single floating-point precision:

| Matrix Dimensions | Neanderthal | jBLAS | Vectorz | Neanderthal vs jBLAS | Neanderthal vs Vectorz |
| -----------------:| -----------:| -----:| -------:| --------------------:| ----------------------:|
| 2x2 | 117.80911 ns | 300.69979 ns | 57.37400 ns | 2.55 | 0.49 |
| 4x4 | 142.97028 ns | 328.31548 ns | 132.33028 ns | 2.30 | 0.93 |
| 8x8 | 330.86712 ns | 539.83386 ns | 613.88615 ns | 1.63 | 1.86 |
| 16x16 | 1.10251 µs | 1.58609 µs | 3.79212 µs | 1.44 | 3.44 |
| 32x32 | 6.40519 µs | 9.03898 µs | 25.82390 µs | 1.41 | 4.03 |
| 64x64 | 47.28739 µs | 66.96849 µs | 210.81286 µs | 1.42 | 4.46 |
| 128x128 | 191.30265 µs | 299.16114 µs | 1.55666 ms | 1.56 | 8.14 |
| 256x256 | 638.76006 µs | 1.29501 ms | 12.04184 ms | 2.03 | 18.85 |
| 512x512 | 4.78386 ms | 8.53300 ms | 94.65790 ms | 1.78 | 19.79 |
| 1024x1024 | 38.00556 ms | 68.52859 ms | 751.53860 ms | 1.80 | 19.77 |
| 2048x2048 | 287.81787 ms | 538.49383 ms | 6.01701 sec | 1.87 | 20.91 |
| 4096x4096 | 2.34824 sec | 4.30298 sec | 45.92363 sec | 1.83 | 19.56 |
| 8192x8192 | 17.88468 sec | 4.55068 min | 6.14534 min | 15.27 | 20.62 |

#### With serial ATLAS, double floating-point precision:

| Matrix Dimensions | Neanderthal | jBLAS | Vectorz | Neanderthal vs jBLAS | Neanderthal vs Vectorz |
| -----------------:| -----------:| -----:| -------:| --------------------:| ----------------------:|
| 2x2 | 113.17721 ns | 6.87139 µs | 57.37400 ns | 60.71 | 0.51 |
| 4x4 | 144.56696 ns | 7.03403 µs | 132.33028 ns | 48.66 | 0.92 |
| 8x8 | 369.70623 ns | 7.23608 µs | 613.88615 ns | 19.57 | 1.66 |
| 16x16 | 1.18290 µs | 8.31470 µs | 3.79212 µs | 7.03 | 3.21 |
| 32x32 | 8.23526 µs | 15.56198 µs | 25.82390 µs | 1.89 | 3.14 |
| 64x64 | 24.23202 µs | 50.65159 µs | 210.81286 µs | 2.09 | 8.70 |
| 128x128 | 164.26852 µs | 305.65059 µs | 1.55666 ms | 1.86 | 9.48 |
| 256x256 | 1.23872 ms | 2.54841 ms | 12.04184 ms | 2.06 | 9.72 |
| 512x512 | 8.99138 ms | 18.93424 ms | 94.65790 ms | 2.11 | 10.53 |
| 1024x1024 | 70.59805 ms | 136.37988 ms | 751.53860 ms | 1.93 | 10.65 |
| 2048x2048 | 543.67433 ms | 1.08081 sec | 6.01701 sec | 1.99 | 11.07 |
| 4096x4096 | 4.28040 sec | 8.58928 sec | 45.92363 sec | 2.01 | 10.73 |
| 8192x8192 | 34.21487 sec | 1.10277 min | 6.14534 min | 1.93 | 10.78 |
