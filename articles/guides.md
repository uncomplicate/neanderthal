---
title: "Neanderthal: Tutorials"
Author: Dragan Djuric
layout: article
---

## Basic setup

* [Getting Started](/articles/getting_started.html) helps you to set up your environment.
* [Hello world](https://github.com/uncomplicate/neanderthal/tree/master/examples/hello-world), a project that helps at the very beginning.

## Presentations (Slides & Videos)

* [EuroClojure 2016 - Clojure is Not Afraid of the GPU](http://2016.euroclojure.org/speakers#ddjuric): Take a look at the [video](https://www.youtube.com/watch?v=bEOOYbscyTs) and [slides](http://dragan.rocks/talks/EuroClojure2016/clojure-is-not-afraid-of-the-gpu.html). Please note that Neanderthal became much faster in later releases than it was in this presentation.
* [Bob Konferenz 2017 - Bayadera: Bayes + Clojure + GPU](http://bobkonf.de/2017/djuric.html): Take a look at the [video](https://www.youtube.com/watch?v=TGxYfi3Vi3s) and [slides](http://dragan.rocks/talks/Bobkonferenz2017/bayadera-bob.html). Bayadera is a cool library that uses Neanderthal.

## General and native engine tutorials

* [Fast, Native Speed, Vector Computations in Clojure](/articles/tutorial_native.html), and the [source code](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_native_test.clj) for this tutorial.
* [Fast Map and Reduce for Primitive Vectors and Matrices](/articles/fast-map-and-reduce-for-primitive-vectors.html), which also comes with [source code](https://github.com/uncomplicate/neanderthal/blob/master/examples/benchmarks/map_reduce.clj) for this tutorial.
* [Neanderthal API Reference](/codox) contains the desrciption of each function, and also comes with mini examples. **There actually is helpful stuff there. Do not skip it!**

## GPU computation tutorials

* [Matrix Computations on the GPU in Clojure (in TFLOPS!)](/articles/tutorial_opencl.html). Proceed to this GPU engine tutorial when you want even more speed ([source code](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_opencl_test.clj)).

* [CUDA and cuBLAS GPU matrices in Clojure](http://dragan.rocks/articles/17/CUDA-and-cuBLAS-GPU-matrices-in-Clojure). The CUDA engine announcement blog post.

## Linear Algebra Tutorials

* [Clojure Linear Algebra Refresher (1) - Vector Spaces](http://dragan.rocks/articles/17/Clojure-Linear-Algebra-Refresher-Vector-Spaces)
* [Clojure Linear Algebra Refresher (2) - Eigenvalues and eigenvectors](http://dragan.rocks/articles/17/Clojure-Linear-Algebra-Refresher-Eigenvalues-and-Eigenvectors)
* [Clojure Linear Algebra Refresher (3) - Matrix Transformations](http://dragan.rocks/articles/17/Clojure-Linear-Algebra-Refresher-Matrix-Transformations)
* [Coding the Matrix in Neanderthal](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal/examples/codingthematrix) contains examples that follow the [Coding the Matrix](http://codingthematrix.com/)
book.

## Internal details and edge cases

* [Neanderthal Tests](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal) show many more details, but without explanations in prose. It will help when you are struggling with an edge case.

# Making sense of legacy BLAS & LAPACK

BLAS (Basic Linear Algebra Subroutines) and LAPACK are mature and de facto standards
for numerical linear algebra. They might seem arcane at first glance,but with a
little exploration, it all have sense and reason.

## Where to find legacy BLAS & LAPACK documentation

When you need more information beyond [Neanderthal API](/codox), these links can help you:

* [BLAS Netlib Homepage](http://netlib.org/blas/)
* [LAPACK Netlib Homepage](http://netlib.org/lapack/)
* [LAPACK Users' Guide](http://www.netlib.org/lapack/lug/)
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
