---
title: "Get Started"
Author: Dragan Djuric
layout: article
---

Neanderthal's default option is to use use native libraries, so it is very important that you do not skip any part of this guide.

## How to Get Started

* Walk through this guide, set up your development environment, and try the examples.
* Familiarize yourself with Neanderthal's [more detailed tutorials](/articles/guides.html) and [API documentation](/codox).

## Usage

This is what you'll be able to do after the [installation](#installation):

First `use` or `require` `uncomplicate.neanderthal.core` and `uncomplicate.neanderthal.native` in your namespace, and you'll be able to call appropriate functions from the Neanderthal library.

```clojure
(ns example
  (:use [uncomplicate.neanderthal core native]))
```

Now you can create neanderthal vectors and matrices, and do computations with them.

Here is how we create two double-precision vectors and compute their dot product:

```clojure
(def x (dv 1 2 3))
(def y (dv 10 20 30))
(dot x y)
```

This is one of the ways to multiply matrices:

```clojure
(def a (dge 3 2 [1 2 3 4 5 6]))
(def b (dge 2 3 [10 20 30 40 50 60]))
(mm a b)
```

## Overview and Features

Neanderthal is a Clojure library for fast matrix and linear algebra computations that supports pluggable engines:

* The **native engine** is based on a highly optimized native [Intel's MKL](http://https://software.intel.com/en-us/intel-mkl) library of [BLAS](http://netlib.org/blas/) and [LAPACK](http://www.netlib.org/lapack/) computation routines (MKL is not open-source, but it is free to use and redistribute since 2016).
* The **GPU engine** is based on OpenCL BLAS routines from [CLBlast](https://github.com/CNugteren/CLBlast) library for even more computational power when needed. It uses [ClojureCL](http://clojurece.uncomplicate.org) and [JOCL](http://jocl.org) libraries. Check out [Uncomplicate ClojureCL](http://clojurecl.uncomplicate.org) if you want to harness the GPU power from Clojure for your own algorithms.

### Implemented Features

* Data structures: double and single precision vectors, dense matrices (GE), and triangular matrices (TR);
* BLAS Level 1, 2, and 3 routines;
* Major LAPACK routines;
* Various Clojure vector and matrix functions (transpositions, submatrices etc.);
* Easy and efficient data mapping and transfer to and from GPUs;
* Fast map, reduce and fold implementations for the provided structures.

### On the TODO List

* CUDA support (in addition to OpenCL support that already works with Nvidia)
* Banded, symmetric, and sparse matrices;
* More LAPACK routines;
* Support for complex numbers;

## Installation

1. Add Neanderthal jars to your classpath ([from the Clojars](https://clojars.org/uncomplicate/neanderthal)).
2. To use the native engine: install Intel's MKL on your system following [Native Engine Requirements](#the-native-library-used-by-neanderthals-native-engine)).
3. To use the GPU engine: install the drivers and an OpenCL platform software provided by the vendor of your graphic card (you probably already have that; see [GPU Engine Requirements](#gpu-drivers-for-the-gpu-engine)).

### With Leiningen

The most straightforward way to include Neanderthal in your project is with Leiningen. Add the following dependency to your `project.clj`, just like in [the Hello World project](https://github.com/uncomplicate/neanderthal/blob/master/examples/hello-world/project.clj):

![](http://clojars.org/uncomplicate/neanderthal/latest-version.svg)

## Requirements

You need at least Java 8.

Neanderthal's data structures are written in Clojure, so many functions work even without native engines. However, you probably need Neanderthal because of its fast BLAS native or GPU engines. Here is how to make sure they are available.

### The native library used by Neanderthal's native engine

* Works on Linux, OS X, and Windows!

* Note: Previous versions of Neanderthal used ATLAS, while from version 0.9.0, it uses Intel's MKL. If you used ATLAS previously, you'll have to install MKL!

* While old versions had different installation guides for each OS, from version 0.9.0, installation on all OSes is more or less the same.

Neanderthal **uses the native Intel MKL library and expects that you make it available on your system, typically as shared xyz.so, xyz.dll, or xyz.dylib files**. Intel MKL is highly optimized for various architectures; its installation comes with many optimized binaries for all supported architectures, that are then selected during runtime according to the hardware at hand. Neanderthal has been built and tested with **Intel MKL 2017**; please make sure that you use a compatible MKL version.

**You do not need to compile or tune anything yourself.**

There are two main steps to how to make MKL available on your system; either:

1. (Optional) Install MKL using a [GUI installer provided by Intel](https://software.intel.com/en-us/intel-mkl) free of charge. In case you use this method, you may [set environment variables as explained in this guide](https://software.intel.com/en-us/node/528500), but it is probably not required, since you do **not** need to compile anything.
2. Put all required binary files (that you installed with MKL installer or copied from wherever you acquired them) in a directory that is available from your `LD_LIBRARY_PATH` or, if you are using Windows, `PATH` (those binary files are available from anyone who installed MKL and have the (free) license to redistribute it with a project).

Either way, Neanderthal does not care how MKL has been provided, as long as it is on the path of your OS. When it comes to distributing the software you build using Neanderthal, I guess the most convenient option is that you include the required MKL binary files in the uberjar or other package that you use to ship the product. Then, it would not require any additional action from your users.

This is the list of MKL files that Neanderthal requires:

* Mandatory MKL libraries (this is what I use on my Intel Core i7 4790k):
  * `libmkl_rt.so`
  * `libmkl_core.so`
  * `libmkl_intel_lp64.so`
  * `libiomp5.so`
  * `libmkl_intel_thread`
  * `libmkl_avx2.so` (if your CPU supports AVX2, which it probably does)
* Optionally, your processor might support additional set of instructions that may require an additional file from MKL. See the MKL guide and browse all available files once you install MKL to discover this. For example, I guess that you might also need `libmkl_avx512.so` if you have a Xeon processor.

Please note that, if you use Windows or OS X, the binary file extensions are not `.so`, but `.dll` and `dylib` respectively.

**Note for OSX users:** MKL installation on my OSX 10.11 placed `libiomp5.dylib` in a different folder than the rest
of the MKL libraries. In such case, copy that file where it is visible, or don't rely on the MKL installation, but select the needed library files and put them somewhere on the `LD_LIBRARY_PATH`. Please also note that on OSX, this environment variable could instead be called `DYLD_LIBRARY_PATH` - you'll have to consult Intel MKL documentation and experiment a bit in such cases.

**Note for Windows users:** MKL installation on my Windows 10 keeps all required `.dll` files in the `<install dir>\redist` folder. The usual folders that keep `.so` and `dylib` on Linux and OSX, keep `.lib` files on Windows - you do not need those.

### GPU drivers for the GPU engine

Everything will magically work (no need to compile anything) on Nvidia, AMD, and Intel's GPUs and CPUs as long as you have appropriate GPU drivers.

Works on Linux, Windows, and OS X!

Follow the [ClojureCL getting started guide](http://clojurecl.uncomplicate.org/articles/getting_started.html) for the links for the GPU platform that you use and more detailed info.

**If you use a pre-2.0 OpenCL platform (Nvidia and/or OS X), you'll have to use `command-queue-1` and/or `with-default-1` from the [ClojureCL's legacy namespace](https://github.com/uncomplicate/clojurecl/blob/master/src/clojure/uncomplicate/clojurecl/legacy.clj) instead of `command-queue` and `with-default` that are used in the examples.**

## Where to Go Next

Hopefully this guide got you started and now you'd like to learn more. I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [Neanderthal API](/codox) for specific details, and feel free to take a glance at [the source](https://github.com/uncomplicate/neanderthal) while you are there.

It is also a good idea to follow [my blog at dragan.rocks](http://dragan.rocks) since I'll write about Neanderthal there.
