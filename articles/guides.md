---
title: "Neanderthal: Guides"
Author: Dragan Djuric
layout: article
---

## Basic setup

* [Getting Started](/articles/getting_started.html) helps you to set up your environment.
* [Hello world](https://github.com/uncomplicate/neanderthal/tree/master/examples/hello-world), a project that helps at the very beginning.

## Books

[New book available for subscription.]("https://aiprobook.com/deep-learning-for-programmers")

<a href="https://aiprobook.com/deep-learning-for-programmers">
<img src="http://aiprobook.com/img/dlfp-cover.png" style="max-width: 100%"/>
</a>

## Presentations (Slides & Videos)

* [ClojuTRE & SmallFP 2018 - Interactive, Functional, GPU Accelerated Programming in Clojure](https://clojutre.org/2018/#dragandjuric): Take a look at the [video](https://www.youtube.com/watch?v=ZVnbNLks2Ow). Neanderthal, CUDA, etc.

* [Bob Konferenz 2017 - Bayadera: Bayes + Clojure + GPU](https://bobkonf.de/2017/djuric.html): Take a look at the [video](https://www.youtube.com/watch?v=TGxYfi3Vi3s) and [slides](https://dragan.rocks/talks/Bobkonferenz2017/bayadera-bob.html). Bayadera is a cool library that uses Neanderthal.

* [EuroClojure 2016 - Clojure is Not Afraid of the GPU](https://2016.euroclojure.org/speakers#ddjuric): Take a look at the [video](https://www.youtube.com/watch?v=bEOOYbscyTs) and [slides](https://dragan.rocks/talks/EuroClojure2016/clojure-is-not-afraid-of-the-gpu.html). Please note that Neanderthal became much faster in later releases than it was in this presentation.

## Performance comparisons with other fast libraries

* [Neanderthal vs ND4J - vol 1 – Native Performance, Java and CPU](https://dragan.rocks/articles/18/Neanderthal-vs-ND4J-vol1)
* [Neanderthal vs ND4J - vol 2 - The Same Native MKL Backend, 1000 x Speedup](https://dragan.rocks/articles/18/Neanderthal-vs-ND4J-vol2).

* [Neanderthal vs ND4J - vol 3 - Clojure Beyond Fast Native MKL Backend](https://dragan.rocks/articles/18/Neanderthal-vs-ND4J-vol3).

* [Neanderthal vs ND4J - vol 4 - Fast Vector Broadcasting in Java, CPU and CUDA](https://dragan.rocks/articles/18/Neanderthal-vs-ND4J-vol4).

* [Neanderthal vs ND4J - vol 5 - Why are native map and reduce up to 100x faster in Clojure?](https://dragan.rocks/articles/18/Neanderthal-vs-ND4J-vol5).

## General and native engine tutorials

* [Fast, Native Speed, Vector Computations in Clojure](/articles/tutorial_native.html), and the [source code](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_native_test.clj) for this tutorial.
* [Fast Map and Reduce for Primitive Vectors and Matrices](/articles/fast-map-and-reduce-for-primitive-vectors.html), which also comes with [source code](https://github.com/uncomplicate/neanderthal/blob/master/examples/benchmarks/map_reduce.clj) for this tutorial.
* [Neanderthal API Reference](/codox) contains the desrciption of each function, and also comes with mini examples. **There actually is helpful stuff there. Do not skip it!**

## Deep Learning From Scratch To GPU

* [Part 0 - Why Bother?](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-0-Why-Bother)

* [Part 1 - Representing Layers and Connections](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-1-Representing-Layers-and-Connections)

* [Part 2 - Bias and Activation Function](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-2-Bias-and-Activation-Function)

* [Part 3 - Fully Connected Inference Layers](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-3-Fully-Connected-Inference-Layers)

* [Part 4 - Increasing Performance with Batch Processing](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-4-Increasing-Performance-with-Batch-Processing)

* [Part 5 - Sharing Memory](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-5-Sharing-Memory)

* [Part 6 - CUDA and OpenCL](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-6-CUDA-and-OpenCL)

* [Part 7 - Learning and Backpropagation](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-7-Learning-and-Backpropagation)

* [Part 8 - The Forward Pass (CUDA, OpenCL, Nvidia, AMD, Intel)](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-8-The-Forward-Pass-CPU-GPU-CUDA-OpenCL-Nvidia-AMD-Intel)

* [Part 9 - The Activation and its Derivative](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-9-The-Activation-and-its-Derivative)

* [Part 10 - The Backward Pass](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-10-The-Backward-Pass-CDU-GPU-CUDA-OpenCL-Nvidia-AMD-Intel)

* [Part 11 - A Simple Neural Network Inference API](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-11-A-Simple-Neural-Network-API)

* [Part 12 - A Simple Neural Network Training API](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-12-A-Simple-Neural-Network-Training-API)

* [Part 13 - Initializing Weights](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-13-Initializing-Weights)

* [Part 14 - Learning a Regression](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-14-Learning-Regression)

* [Part 15 - Weight Decay](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-15-Weight-Decay)

* [Part 16 - Momentum](https://dragan.rocks/articles/19/Deep-Learning-in-Clojure-From-Scratch-to-GPU-16-Momentum)

## GPU computation tutorials

* [Matrix Computations on the GPU in Clojure (in TFLOPS!)](/articles/tutorial_opencl.html). Proceed to this GPU engine tutorial when you want even more speed ([source code](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_opencl_test.clj)).

* [CUDA and cuBLAS GPU matrices in Clojure](https://dragan.rocks/articles/17/CUDA-and-cuBLAS-GPU-matrices-in-Clojure). The CUDA engine announcement blog post.

## Linear Algebra Tutorials

* [Clojure Linear Algebra Refresher (1) - Vector Spaces](https://dragan.rocks/articles/17/Clojure-Linear-Algebra-Refresher-Vector-Spaces)
* [Clojure Linear Algebra Refresher (2) - Eigenvalues and eigenvectors](https://dragan.rocks/articles/17/Clojure-Linear-Algebra-Refresher-Eigenvalues-and-Eigenvectors)
* [Clojure Linear Algebra Refresher (3) - Matrix Transformations](https://dragan.rocks/articles/17/Clojure-Linear-Algebra-Refresher-Matrix-Transformations)
* [Clojure Linear Algebra Refresher (4) - Linear Transformations](https://dragan.rocks/articles/17/Clojure-Linear-Algebra-Refresher-Linear-Transformations)
* [Coding the Matrix in Neanderthal](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal/examples/codingthematrix) contains examples that follow the [Coding the Matrix](https://codingthematrix.com/)
book.

## Clojure Numerics

* [Clojure Numerics, Part 1 - Use Matrices Efficiently](https://dragan.rocks/articles/17/Clojure-Numerics-1-Use-Matrices-Efficiently)
* [Clojure Numerics, Part 2 - General Linear Systems and LU Factorization](https://dragan.rocks/articles/17/Clojure-Numerics-2-General-Linear-Systems-and-LU-Factorization)
* [Clojure Numerics, Part 3 - Special Linear Systems and Cholesky Factorization](https://dragan.rocks/articles/17/Clojure-Numerics-3-Special-Linear-Systems-and-Cholesky-Factorization)
* [Clojure Numerics, Part 4 - Singular Value Decomposition (SVD)](https://dragan.rocks/articles/17/Clojure-Numerics-4-Singular-Value-Decomposition-SVD)
* [Clojure Numerics, Part 5 - Orthogonalization and Least Squares](https://dragan.rocks/articles/17/Clojure-Numerics-5-Orthogonalization-and-Least-Squares)
* [Clojure Numerics, Part 6 - More Linear Algebra Fun with Least Squares](https://dragan.rocks/articles/17/Clojure-Numerics-6-More-Linear-Algebra-Fun-with-Least-Squares)

## Internal details and edge cases

* [Neanderthal Tests](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal) show many more details, without explanations in prose. This will help when you are struggling with an edge case.

# Making sense of legacy BLAS & LAPACK

BLAS (Basic Linear Algebra Subroutines) and LAPACK are mature and de facto standards
for numerical linear algebra. They might seem arcane at first glance,but with a
little exploration, it all have sense and reason.

## Where to find legacy BLAS & LAPACK documentation

When you need more information beyond [Neanderthal API](/codox), these links can help you:

* [BLAS Netlib Homepage](https://netlib.org/blas/)
* [LAPACK Netlib Homepage](https://netlib.org/lapack/)
* [LAPACK Users' Guide](https://www.netlib.org/lapack/lug/)
* [Intel MKL Developer Reference](https://software.intel.com/en-us/mkl-reference-manual-for-c)

A more detailed doc for each subroutine is available at these links.

## Naming conventions (briefly)

You see a procedure named `DGEMM` in the aforementioned docs and you are completely baffled. Here is how to decipher it:

* D is for Double precision, meaning the function works with doubles
* GE is for GEneral dense matrix, meaning we store all these doubles and not using any clever tricks for special matrix structures like symmetric, triangular etc.
* MM is the operation, it stands for Matrix Multiply.

Generally, Neanderthal will try to abstract away the primitive type (D) and the actual structure (GE) so you can
use methods like `mm` to multiply two matrices.

It is a good idea to familiarize yourself with [BLAS Naming Scheme](https://software.intel.com/en-us/node/520726). It will help you understand how to efficiently use Neanderthal functions and where to look for a function that does what you need to do.
