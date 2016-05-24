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

* The **native engine** is based on a highly optimized native [Automatically Tuned Linear Algebra Software (ATLAS)](http://math-atlas.sourceforge.net/) library of [BLAS](http://netlib.org/blas/) and [LAPACK](http://www.netlib.org/lapack/) computation routines.
* The **GPU engine** is based on custom OpenCL kernels for BLAS routines for even more computational power when needed. That one is written in Clojure (except the kernels themselves)!
Check out [Uncomplicate ClojureCL](http://clojurecl.uncomplicate.org) if you want to harness the GPU power from Clojure for your own algorithms.

### Implemented Features

* Data structures: double and single precision vector, double and single precision general dense matrix (GE);
* BLAS Level 1, 2, and 3 routines;
* Various Clojure vector and matrix functions (transpositions, submatrices etc.);
* Fast map, reduce and fold implementations for the provided structures.

### On the TODO List

* LAPACK routines;
* Banded, symmetric, triangular, and sparse matrices;
* Support for complex numbers;

## Installation

1. Add Neanderthal jars to your classpath ([from the Clojars](https://clojars.org/uncomplicate/neanderthal)).
2. To use the native engine: install ATLAS on your system following the [ATLAS Installation Guide](http://math-atlas.sourceforge.net/atlas_install/atlas_install.html). (see [Requirements](#requirements))
3. To use the GPU engine: install the drivers (you probably already have that) and an OpenCL platform software provided by the vendor of your graphic card ([as in the ClojureCL getting started guide](http://clojurecl.uncomplicate.org/articles/getting_started.html)).

### With Leiningen

The most straightforward way to include Neanderthal in your project is with Leiningen. Add the following dependency to your `project.clj`, just like in [the Hello World project](https://github.com/uncomplicate/neanderthal/blob/master/examples/hello-world/project.clj):

![](http://clojars.org/uncomplicate/neanderthal/latest-version.svg)

## Requirements

Neanderthal's data structures are written in Clojure, so many functions work out of the box. However, you probably need Neanderthal because of its fast BLAS native and/or GPU engines.

### The native library used by Neanderthal's native engine

#### Mac OS X

**Works out of the box**. You should have Apple's XCode that comes with Accelerate framework and that's it - no need to fiddle with ATLAS, and you get Apple's highly tuned BLAS engine.

#### Linux - optimized

Neanderthal **uses the native ATLAS library and expects that you make it available on your system, typically as a shared libatlas.so** ATLAS is highly optimized for various architectures - if you want top performance **you have to build ATLAS from the source**. Do not worry, ATLAS comes with automatic autotools build script, and a [detailed configuration and installation guide](http://math-atlas.sourceforge.net/atlas_install/atlas_install.html).

If you do not follow this procedure, and use a pre-packaged ATLAS provided by your package manager (available in most distributions), you will probably get degraded performance compared to a properly installed ATLAS.

Either way, Neanderthal does not care how ATLAS is provided, as long it is in the system path an was compiled as a shared library. It can even be packed in a jar if that's convenient, and I could make some steps in the future to make
the "compile it and install ATLAS by yourself" step optional. But, I do not recommend it, other than as a convenience for people who are scared of the terminal and C tools.

This is how I installed it on Arch Linux:

* I had to have gcc (installed by default) and gcc-fortran packages.
* I followed the aforementioned atlas build instructions to the letter. The only critical addition is to add `--shared` flag (explained in the details there, but not a default).
* I had to disable CPU throttling with this command in the shell: `cpupower frequency-set -g performance`
* I had to create a symlink `libatlas.so` in my `/usr/lib`, that points to 'libsatlas.so' (serial)
or 'libtatlas.so' (parallel) atlas shared binary created by the build script.

That should be all, but YMMV, depending on your hardware and OS installation.

#### Linux - unoptimized, but easy way (NOT recommended)

Use atlas build provided by your package manager. Something like:

``` shell
sudo pacman -Suy atlas-lapack
```
or your distribution's equivalent. It is fine as an easy way to get started, but **does not offer full performance**.

#### Windows

I do not have a copy of Windows, so I do not provide pre-build library for Windows. ATLAS needs [special build instructions for Windows](http://math-atlas.sourceforge.net/atlas_install/node50.html) in addition to  general [detailed configuration and installation guide](http://math-atlas.sourceforge.net/atlas_install/atlas_install.html)

**If you know your way around gcc and MinGW on Windows, and are willing to help users of that operating system by providing the convenient binaries, please [contact me](/articles/community.html).**

### GPU drivers for the GPU engine

Everything will magically work (no need to compile anything) on Nvidia, AMD, and Intel's GPUs and CPUs as long as you have appropriate GPU drivers.

Works on Linux, Windows, and OS X!

Follow the [ClojureCL getting started guide](http://clojurecl.uncomplicate.org/articles/getting_started.html) for the links for the GPU platform that you use and more detailed info.

** If you use a pre-2.0 OpenCL platform (Nvidia and/or OS X), you'll have to use `command-queue-1` and/or `with-default-1` from the [ClojureCL's legacy namespace](https://github.com/uncomplicate/clojurecl/blob/master/src/clojure/uncomplicate/clojurecl/legacy.clj) instead of `command-queue` and `with-default` that are used in the examples.**

## Where to Go Next

Hopefully this guide got you started and now you'd like to learn more. I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [Neanderthal API](/codox) for specific details, and feel free to take a gander at [the source](https://github.com/uncomplicate/neanderthal) while you are there.
