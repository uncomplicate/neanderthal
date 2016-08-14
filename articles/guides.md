---
title: "Neanderthal: Guides"
Author: Dragan Djuric
layout: article
---

# Neanderthal Tutorials

## Basic setup

* [Getting Started](/articles/getting_started.html) helps you to set up your environment.
* [Hello world](https://github.com/uncomplicate/neanderthal/tree/master/examples/hello-world), a project that can be helpful for quick start.

## General and native engine tutorials

* [Fast, Native Speed, Vector Computations in Clojure](/articles/tutorial_native.html), and the [source code](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_native_test.clj) for this tutorial.
* [Fast Map and Reduce for Primitive Vectors and Matrices](/articles/fast-map-and-reduce-for-primitive-vectors.html), which also comes with [source code](https://github.com/uncomplicate/neanderthal/blob/master/examples/benchmarks/map_reduce.clj) for this tutorial.
* [Neanderthal API](/codox) contains the desrciption of each function, and also comes with mini examples.

## GPU computation tutorials

* [Matrix Computations on the GPU in Clojure (in TFLOPS!)](/articles/tutorial_opencl.html). Proceed to this GPU engine tutorial when you want even more speed ([source code](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_opencl_test.clj)).

## Further examples

* [Coding the Matrix in Neanderthal](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal/examples/codingthematrix) contains examples that follow the [Coding the Matrix](http://codingthematrix.com/)
book.

## Internal details and edge cases

* [Neanderthal Tests](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal) show some more details, but without explanations in prose. It might help when you are struggling with an edge case.

# Making sense of legacy BLAS & LAPACK

BLAS (Basic Linear Algebra Subroutines) and LAPACK are mature and de facto standards
for numerical linear algebra. They might seem arcane at first glance,but with a
little exploration, it all have sense and reason.

## Where to find legacy BLAS & LAPACK documentation

When you need more information beyond [Neanderthal API](/codox), these links can help you:

* [BLAS Netlib Homepage](http://netlib.org/blas/)
* [LAPACK Netlib Homepage](http://netlib.org/lapack/)
* [LAPACK Users' Guide](http://www.netlib.org/lapack/lug/)
* [NEC MathKeisan man pages](http://www.mathkeisan.com/UsersGuide/man.cfm)

A more detailed doc for each subroutine is available at these links.

### When that is not enough

If that is not enough, you should look out for guides for mature scientific
software libraries provided by the companies like IBM, NEC etc. These might shed
some light, but I hope you won't need to go that far, since I will try to hide
such details behind Neanderthal.

## Naming conventions (briefly)

You see a procedure named `DGEMM` in the aforementioned docs and you are completely baffled. Here is how to decipher it:

* D is for Double precision, meaning the function works with doubles
* GE is for GEneral dense matrix, meaning we store all these doubles and not using any clever tricks for special matrix structures like symmetric, triangular etc.
* MM is the operation, it stands for Matrix Multiply.

Generally, Neanderthal will try to abstract away the primitive type (D) and the actual structure (GE) so you can
use methods like `mm` to multiply two matrices.

###  [BLAS & LAPACK Naming Scheme](http://www.netlib.org/lapack/lug/node24.html)
