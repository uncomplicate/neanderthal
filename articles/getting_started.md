---
title: "Get Started"
Author: Dragan Djuric
layout: article
---

Neanderthal's default option is to use native libraries, so it is very
important that you do not skip any part of this guide.

## How to Get Started

* Walk through this guide, set up your development environment, and try the examples.
* Familiarize yourself with Neanderthal's [more detailed tutorials](/articles/guides.html) and [API documentation](/codox).

## Usage

First `use` or `require` `uncomplicate.neanderthal.core` and `uncomplicate.neanderthal.native`
in your namespace. You'll then be able to call appropriate functions from the Neanderthal library.

```clojure
(ns example
  (:use [uncomplicate.neanderthal core native]))
```

Now you can create Neanderthal vectors and matrices and do computations with them.

Here is how we create two double-precision vectors and compute their dot product:

```clojure
(def x (dv 1 2 3))
(def y (dv 10 20 30))
(dot x y)
```

This is one of the ways to multiply matrices:

```clojure
(def a (dge 3 2 [1 2 3 4 5 6))
(def b (dge 2 3 [10 20 30 40 50 60]))
(mm a b)
```


## Overview and Features

Neanderthal is a Clojure library for fast matrix and linear algebra computations
that supports pluggable engines:

* The **native engine** is based on a highly optimized native [Automatically Tuned Linear Algebra Software (ATLAS)](http://math-atlas.sourceforge.net/)
library of [BLAS](http://netlib.org/blas/) and [LAPACK](http://www.netlib.org/lapack/)
computation routines.
* The **GPU engine** is based on custom OpenCL kernels for BLAS routines for even
more computational power when needed. That one is written in Clojure (except the kernels themselves)!
Check out [Uncomplicate ClojureCL](http://clojurecl.uncomplicate.org) if you want to
harness GPU power using Clojure for your own algorithms.

### Implemented Features

* Data structures: double and single precision vectors, double and single precision
general dense matrices (GE);
* BLAS Level 1, 2, and 3 routines;
* Various Clojure vector and matrix functions (transpositions, submatrices etc.);
* Fast map, reduce and fold implementations for the provided structures.

### On the TODO List

* LAPACK routines;
* Banded, symmetric, triangular, and sparse matrices;
* Support for complex numbers

## Installation

1. Add Neanderthal jars to your classpath ([from the Clojars](clojars.org/uncomplicate/neanderthal)).
2. To use the native engine: install ATLAS on your system following the [ATLAS Installation Guide](http://math-atlas.sourceforge.net/atlas_install/atlas_install.html). (see [Requirements](#requirements))
3. To use the GPU engine: install the drivers (you probably already have those) and an
OpenCL platform software provided by the vendor of your graphics card ([as in the ClojureCL getting started guide](http://clojurecl.uncomplicate.org/articles/getting_started.html)).

### With Leiningen

The most straightforward way to include Neanderthal in your project is with Leiningen. Add the following dependencies to your `project.clj`, just like in [the Hello World project](https://github.com/uncomplicate/neanderthal/blob/master/examples/hello-world/project.clj):

```clojure
[uncomplicate/neanderthal "0.4.0"]
[uncomplicate/neanderthal-atlas "0.1.0" :classifier "amd64-Linux-gpp-jni"]
```

Replace `amd64-Linux-gpp-jni` with your system's architecture and OS classifier.
If you are not sure what exactly to write, check `(System/getProperty "os.arch")`
and `(System/getProperty "os.name")`.

**MacOSX** will be available in Clojars in the next release. In the meantime,
you should download [neanderthal-atlas-0.1.0-x86_64-MacOSX-gpp-jni.jar](https://mega.nz/#!uwB10LDY!Mb_oKJf8X-C9KBQ1haNRVnKcF55cedNYYUQeie2i1HI) and put it in your `.m2` directory.

I will always provide at least the Linux build for the native jar of the library in Clojars.
If the library for your OS is not in Clojars, checkout [neanderthal-atlas](https://github.com/uncomplicate/neanderthal-atlas)
source and build it with maven using `mvn install`.
The build is fully automatic if you have gcc and other related gnu tools.
If you can successfully build ATLAS, you already have all the necessary tools.

### Requirements

Neanderthal is a Clojure library packed in two `jar` files, distributed through
[Clojars](http://clojars.org). One is a pure Clojure library that you will use
directly, and the other contains native JNI bindings for the ATLAS engine
for a specific operating system. They follow [maven](http://www.maven.org)
(and [leiningen](http://www.leiningen.org)) naming conventions:

* Pure Clojure lib: `groupId` is `uncomplicate` and `artifactId` is `neanderthal`.
* Native JNI lib: `groupId` is `uncomplicate` and `artifactId` is `neanderthal-atlas` with a classifier for your operating system and architecture, e.g. `amd64-Linux-gpp-jni`.

### The native library used by Neanderthal's native engine

This part is relevant for Linux and Windows. as Mac OS X ships an optimized BLAS engine
out of the box when you install Xcode (or whatever is the current name of Apple's
huge development package).

Neanderthal **uses the native ATLAS library and expects that you make it
available on your system, typically as a shared so, dll, or dylib!** ATLAS is
highly optimized for various architectures - if you want top performance
**you have to build ATLAS from the source**. Do not worry, ATLAS comes with
automatic autotools build script, and a [detailed configuration and installation guide](http://math-atlas.sourceforge.net/atlas_install/atlas_install.html).
If you do not follow this procedure and use a pre-packaged ATLAS provided by
your system (if it exists), you will probably get degraded performance compared
to a properly installed ATLAS.

Either way, Neanderthal does not care how ATLAS is provided as long it is in
the system path and was compiled as a shared library. It can even be packed in
a jar if that's convenient, and I could add some steps in the future to make
the "compile and install ATLAS by yourself" step optional. But I do not
recommend it, other than as a convenience for people who are scared of the
terminal and C tools.

**If you know your way around gcc on OS X, or gcc and MinGW on Windows, and are
willing to help users of those operating systems by providing the convenient
binaries, please [contact me](/articles/community.html).**

### GPU drivers for the GPU engine

Everything will magically work (no need to compile anything) as long as you
have the appropriate hardware (a GPU that supports OpenCL 2.0, which in 2015
means a newer AMD Radeon or FirePro) and install the appropriate drivers and
OpenCL platform which you can obtain from the card vendor's website.
Kernels supporting pre-OpenCL 2.0 and optimized for Nvidia are planned for later.

Follow the [ClojureCL getting started guide](http://clojurecl.uncomplicate.org/articles/getting_started.html)
for links for the GPU platform that you use and more detailed info.

## Where to go next

Hopefully this guide got you started and now you'd like to learn more. I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [Neanderthal API](/codox) for specific details, and feel free to take a gander at [the source](https://github.com/uncomplicate/neanderthal) while you are there.
