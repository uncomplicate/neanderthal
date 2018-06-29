---
title: "Benchmarks"
Author: Dragan Djuric
layout: article
---

## Matrix Multiplication: Neanderthal vs working core.matrix implementations

The focus of this article is Neanderthal's performance on the CPU. Since Neanderthal is a [Clojure](https://clojure.org) library, I compare it to the implementations of [core.matrix](https://github.com/mikera/core.matrix). Neanderthal is also as fast or faster than the best Java and Scala native BLAS wrappers, but I'll leave detailed reports to others. For comparisons of Matrix APIs, or GPU performance, see [the tutorials](guides) ([more than 1500x speedup](tutorial_opencl) for matrix multiplication of very large matrices compared to core.matrix's flagship Vectorz library).

### TL;DR Results

* Neanderthal seems to be the fastest library available on the Java platform. Read detailed comparisons with ND4J: [Neanderthal vs. ND4J – Native Performance, Java and CPU](https://dragan.rocks/articles/18/Neanderthal-vs-ND4J-vol1), [Neanderthal vs ND4J - vol 2 - The Same Native MKL Backend, 1000 x Speedup](https://dragan.rocks/articles/18/Neanderthal-vs-ND4J-vol2), and * [Neanderthal vs ND4J - vol 3 - Clojure Beyond Fast Native MKL Backend](https://dragan.rocks/articles/18/Neanderthal-vs-ND4J-vol3)..

* **Even for very small matrices**, Neanderthal is faster than core.matrix's [Vectorz](https://github.com/mikera/vectorz) (except for matrices smaller than 6x6, and even that is easily fixable by adding a naive Clojure operations for tiny structures). For large matrices, Neanderthal is **much** faster than  Vectorz: **more than 100x faster** (floats) and 50x faster (doubles).

* Neanderthal is **several times faster** than the jBlas-based multithreaded [Clatrix](https://github.com/tel/clatrix) implementation of core.matrix. That does not take into account that [Clatrix has been abandoned](https://github.com/tel/clatrix/issues/62#issuecomment-71420404) by the original author and barely supports core.matrix API, and that Neanderhtal also offers many other features ([GPU computing](tutorial_opencl) being one of them relevant to raw speed).


### What is Being Measured

Neanderthal uses [Intel's Math Kernel Library (MKL)](https://en.wikipedia.org/wiki/Math_Kernel_Library) and adds almost no overhead - it is expected that it will be as fast as your MKL installation (which is the fastest and most featureful thing around).

I will concentrate on the performance of matrix-matrix multiplications, since it is the typical representative of heavy operations O(n^3) (BLAS 3), and is the most telling measure of how well a library performs. Neanderthal takes great care to be as light as possible, and does not use data copying, so even linear operations, vector (BLAS 1) and quadratic matrix-vector (BLAS 2) are faster than in pure Java.

I compared Neanderthal with:

* [Vectorz](https://github.com/mikera/vectorz) - a pure Java/Clojure matrix library. It should be fast for very small matrices, but much slower for large matrices. This is practically the only core.matrix implementation that works and is maintained.

* [Clatrix](https://github.com/tel/clatrix)(a wrapper for [jBLAS](https://mikiobraun.github.io/jblas/)) - an easy to install library that uses a native optimized ATLAS BLAS implementation, and is the most popular and the easiest to install native Java matrix library. It is fast for large matrices, but slow for small matrices. Clatrix works with basic core.matrix API, but has been abandoned by the author some years ago. Issues are fixed very scarcely, and it seems that a considerable portion is broken wrt core.matrix API.

These two libraries are a good representation of the state of matrix computations in Java/Clojure with native (jBLAS) and pure Java (Vectorz) libraries. They also back the most popular Clojure matrix library, [core.matrix](https://github.com/mikera/core.matrix). The core.matrix Wiki currently lists a handful of other implementations, but those haven't seen much work beyond initial exploration and abandonment a few months later, so I am not taking them into account here.

### Results

The code of the benchmark is [here](https://github.com/uncomplicate/neanderthal/blob/master/examples/benchmarks/src/benchmarks/core.clj). The measurements have been done with the help of the [Criterium](https://github.com/hugoduncan/criterium) benchmarking library. On my Intel Core i7 4790k with 32GB of RAM, running Arch Linux, with Neanderthal 0.9.0 calling MKL 2017, and core.matrix calling Clatrix 0.5.0 (which calls jBlas 1.2.3) and Vectorz 0.58.0, the results are:

#### Single precision floating point:

Neanderthal and Clatrix run on 4 cores, Vectorz doesn't have parallelization.

| Matrix Dimensions | Neanderthal | Clatrix | Vectorz | Neanderthal vs Clatrix | Neanderthal vs Vectorz |
| -----------------:| -----------:| -----:| -------:| --------------------:| ----------------------:|
| 2x2 | 232.36 ns | 762.32 ns |  61.36 ns | 3.28 | 0.26 |
| 4x4 | 237.72 ns | 815.51 ns | 129.34 ns | 3.43 | 0.54 |
| 8x8 | 253.22 ns | 880.31 ns | 568.02 ns | 3.48 | 2.24 |
| 16x16 | 372.30 ns |   1.44 µs |   3.45 µs | 3.85 | 9.27 |
| 32x32 | 903.14 ns |   5.45 µs |  23.44 µs | 6.04 | 25.96 |
| 64x64 |   2.80 µs |  17.96 µs | 218.64 µs | 6.43 | 78.21 |
| 128x128 |  16.30 µs |  79.04 µs |   1.55 ms | 4.85 | 94.85 |
| 256x256 | 126.25 µs | 477.62 µs |  12.28 ms | 3.78 | 97.24 |
| 512x512 |   1.07 ms |   4.44 ms |  96.94 ms | 4.13 | 90.21 |
| 1024x1024 |   7.93 ms |  39.36 ms | 778.46 ms | 4.96 | 98.12 |
| 2048x2048 |  57.47 ms | 154.38 ms |   6.22 sec | 2.69 | 108.16 |
| 4096x4096 | 470.12 ms |   1.06 sec |  50.06 sec | 2.26 | 106.49 |
| 8192x8192 |   3.76 sec |   9.24 sec |   6.68 min | 2.46 | 106.56 |

#### Double precision floating point:

Neanderthal and Clatrix run on 4 cores, Vectorz doesn't have parallelization.

| Matrix Dimensions | Neanderthal | Clatrix | Vectorz | Neanderthal vs Clatrix | Neanderthal vs Vectorz |
| -----------------:| -----------:| -----:| -------:| --------------------:| ----------------------:|
| 2x2 | 228.66 ns | 762.32 ns |  61.36 ns | 3.33 | 0.27 |
| 4x4 | 229.29 ns | 815.51 ns | 129.34 ns | 3.56 | 0.56 |
| 8x8 | 263.84 ns | 880.31 ns | 568.02 ns | 3.34 | 2.15 |
| 16x16 | 428.80 ns |   1.44 µs |   3.45 µs | 3.35 | 8.05 |
| 32x32 |   1.52 µs |   5.45 µs |  23.44 µs | 3.60 | 15.47 |
| 64x64 |   6.39 µs |  17.96 µs | 218.64 µs | 2.81 | 34.22 |
| 128x128 |  43.68 µs |  79.04 µs |   1.55 ms | 1.81 | 35.39 |
| 256x256 | 331.98 µs | 477.62 µs |  12.28 ms | 1.44 | 36.98 |
| 512x512 |   3.55 ms |   4.44 ms |  96.94 ms | 1.25 | 27.28 |
| 1024x1024 |  20.05 ms |  39.36 ms | 778.46 ms | 1.96 | 38.83 |
| 2048x2048 | 152.75 ms | 154.38 ms |   6.22 sec | 1.01 | 40.70 |
| 4096x4096 | 999.37 ms |   1.06 sec |  50.06 sec | 1.06 | 50.09 |
| 8192x8192 |   8.84 sec |   9.24 sec |   6.68 min | 1.05 | 45.34 |


#### Single precision floating point (vs jBlas single precision):

Neanderthal and jBlas run on 4 cores, Vectorz doesn't have parallelization.

Since Clatrix does not support single-precision floating point numbers, I did this comparison with jBlas
directly for reference (Neanderthal is still considerably faster :), *but keep in mind that you can't use that from core.matrix*.

| Matrix Dimensions | Neanderthal | jBLAS | Vectorz | Neanderthal vs jBLAS | Neanderthal vs Vectorz |
| -----------------:| -----------:| -----:| -------:| --------------------:| ----------------------:|
| 2x2 | 232.36 ns | 362.00 ns |  61.36 ns | 1.56 | 0.26 |
| 4x4 | 237.72 ns | 369.99 ns | 129.34 ns | 1.56 | 0.54 |
| 8x8 | 253.22 ns | 476.57 ns | 568.02 ns | 1.88 | 2.24 |
| 16x16 | 372.30 ns | 598.43 ns |   3.45 µs | 1.61 | 9.27 |
| 32x32 | 903.14 ns |   1.37 µs |  23.44 µs | 1.52 | 25.96 |
| 64x64 |   2.80 µs |   7.52 µs | 218.64 µs | 2.69 | 78.21 |
| 128x128 |  16.30 µs |  31.48 µs |   1.55 ms | 1.93 | 94.85 |
| 256x256 | 126.25 µs | 191.15 µs |  12.28 ms | 1.51 | 97.24 |
| 512x512 |   1.07 ms |   1.25 ms |  96.94 ms | 1.16 | 90.21 |
| 1024x1024 |   7.93 ms |  10.63 ms | 778.46 ms | 1.34 | 98.12 |
| 2048x2048 |  57.47 ms | 104.95 ms |   6.22 sec | 1.83 | 108.16 |
| 4096x4096 | 470.12 ms | 568.46 ms |  50.06 sec | 1.21 | 106.49 |
| 8192x8192 |   3.76 sec |   4.85 sec |   6.68 min | 1.29 | 106.56 |
