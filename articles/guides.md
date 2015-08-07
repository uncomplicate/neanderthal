---
title: "Guides"
Author: Dragan Djuric
layout: article
---

# Neanderthal Tutorials

First, set up your environment, as described in the [Getting Started](getting_started.html) guide. [Hello world](https://github.com/uncomplicate/neanderthal/tree/master/examples/hello-world) project can also help you with leiningen details.

## General and native engine tutorials

Then, proceed to the native engine tutorial: [Fast, Native Speed, Vector Computations in Clojure](/articles/tutorial_native.html)
([working code](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_native_test.clj)).

Of course, be sure to check out the [Neanderthal API](/codox).

## GPU computation tutorials

Want even more speed? - Proceed to the GPU engine tutorial:
[Matrix Computations on GPU in Clojure (in TFLOPS!)](/articles/tutorial_opencl.html)
([working code](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_opencl_test.clj)).


## Further examples

Some examples that follow the [Coding the Matrix](http://codingthematrix.com/)
book can be found at [Coding the Matrix in Neanderthal](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal/examples/codingthematrix).

## Internal details and edge cases

A bit more details, but without explanations in prose, can be found in
[Neanderthal Tests](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal).
It might help when you are struggling with an edge case.

# Making sense of legacy BLAS & LAPACK

BLAS (Basic Linear Algebra Subroutines) and LAPACK are mature and de facto standards
for numerical linear algebra. They might seem arcane at first glance,but with a
little exprolation, it all have sense and reason.

## Where to find legacy BLAS & LAPACK documentation

BLAS docs reside at the [BLAS Netlib Homepage](http://netlib.org/blas/).

LAPACK docs can be found at [LAPACK Netlib Homepage](http://netlib.org/lapack/).
They include a free [LAPACK Users' Guide](http://www.netlib.org/lapack/lug/).

A more detailed doc for each subroutine is available at
[NEC MathKeisan man pages](http://www.mathkeisan.com/UsersGuide/man.cfm).

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

The complete naming scheme is here: [BLAS & LAPACK Naming Scheme](http://www.netlib.org/lapack/lug/node24.html).
