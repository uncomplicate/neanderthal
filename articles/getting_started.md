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

## Leiningen

The most straightforward way to include Neanderthal in your project is with Leiningen. Add the following dependency to your `project.clj`, just like in **[the Hello World project](https://github.com/uncomplicate/neanderthal/blob/master/examples/hello-world/project.clj)**:

![](https://clojars.org/uncomplicate/neanderthal/latest-version.svg)

Add a MKL distribution jar `[org.bytedeco/mkl-platform-redist "2020.3-1.5.4"]` as your project's dependency.

Neanderhtal will use the native CPU MKL binaries from that jar automatically, so you don't need to do anything else. If the jar is not present, Neanderthal will expect you to have a system-wide MKL installation as explained in [Native Engine Requirements](#the-native-library-used-by-neanderthals-native-engine-optional).33 **Note: MKL distribution size is 750 MB!** Lein will download it the first time you include it, which might take some time, so it's a good idea to run `lein deps` and wait each time you update the version.

## Overview and Features

Neanderthal is a Clojure library for fast matrix and linear algebra computations that supports pluggable engines:

* The **native engine** is based on a highly optimized native [Intel's MKL](https://https://software.intel.com/en-us/intel-mkl) library of [BLAS](https://netlib.org/blas/) and [LAPACK](https://www.netlib.org/lapack/) computation routines (MKL is not open-source, but it is free to use and redistribute since 2016).
* The **CUDA GPU engine** is based on cuBLAS and supports all modern Nvidia GPUs. It uses [ClojureCUDA](https://clojurecuda.uncomplicate.org) and [JCuda](https://jcuda
3.org) libraries. Check out [Uncomplicate ClojureCUDA](https://clojurecuda.uncomplicate.org).
* The **OpenCL GPU engine** is based on OpenCL BLAS routines from [CLBlast](https://github.com/CNugteren/CLBlast) library for even more computational power when needed. It uses [ClojureCL](https://clojurecl.uncomplicate.org) and [JOCL](https://jocl.org) libraries. Check out [Uncomplicate ClojureCL](https://clojurecl.uncomplicate.org).

### Implemented Features

* Data structures: double and single precision vectors, dense matrices (GE), triangular matrices (TR), symmetric matrices (SY), banded, diagonal, etc.;
* BLAS Level 1, 2, and 3 routines;
* Lots of (but not all) LAPACK routines;
* Various Clojure vector and matrix functions (transpositions, submatrices etc.);
* Easy and efficient data mapping and transfer to and from GPUs;
* Fast map, reduce and fold implementations for the provided structures.
* OpenCL GPU support
* CUDA GPU support

### On the TODO List

* ~"Tensors"~: This is already available in Deep Diamond!
* Sparse matrices;
* Support for complex numbers;

## Requirements

You need at least Java 8.

If you are running on Java 9 or higher, you need to enable the `java.base` module. Add this to your JVM options (:jvm-opts in leiningen): `"--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"`.

Neanderthal's data structures are written in Clojure, so many functions work even without native engines. However, you probably need Neanderthal because of its fast BLAS native or GPU engines. Here is how to make sure they are available.

### GPU drivers for the OpenCL GPU engine

Everything will magically work (no need to compile anything) on Nvidia, AMD, and Intel's GPUs and CPUs as long as you have appropriate GPU drivers.

Works on Linux, Windows, and OS X!

Follow the [ClojureCL getting started guide](https://clojurecl.uncomplicate.org/articles/getting_started.html) for the links for the GPU platform that you use and more detailed info.

**If you use a pre-2.0 OpenCL platform (Nvidia and/or OS X), you'll have to use `command-queue-1` and/or `with-default-1` instead of `command-queue` and `with-default`.**

### GPU drivers for the CUDA GPU engine

Everything will magically work (no need to compile anything) on Nvidia, provided that you **installed the latest Nvidia's CUDA Toolkit**.

Follow the [ClojureCUDA getting started guide](https://clojurecuda.uncomplicate.org/articles/getting_started.html) for the links for the GPU platform that you use and more detailed info.

*macOS* doesn't support CUDA 11 and higher (and Apple hasn't shipped Nvidia GPUs since 2014 anyway).

### The native library used by Neanderthal's native engine (Optional)

**The following is not needed if you include [org.bytedeco/mkl-platform-redist "2020.3-1.5.4"] dependency.**

This section deals with system-wide MKL installation.

* Works on Linux, OS X, and Windows!

Neanderthal **uses the native [Intel MKL](https://software.intel.com/en-us/mkl) library and expects that you make it available on your system, typically as shared xyz.so, xyz.dll, or xyz.dylib files**. [Intel MKL](https://software.intel.com/en-us/mkl) is highly optimized for various architectures; its installation comes with many optimized binaries for all supported architectures, that are then selected during runtime according to the hardware at hand.

**You do not need to compile or tune anything yourself.**

These are alternative ways to make MKL available on your system globally; either:

1. Install it through a package manager it that is available on your system. I use Arch Linux (`pacman -Syu intel-mkl`), and I believe that many other Linux distributions now ship MKL in their repositories..

OR:

1. (Optionally) Install MKL using a [GUI installer provided by Intel](https://software.intel.com/en-us/intel-mkl) free of charge. In case you use this method, you may [set environment variables as explained in this guide](https://software.intel.com/en-us/node/528500), but it is probably not required, since you do **not** need to compile anything.
2. Put all required binary files (that you installed with MKL installer or copied from wherever you acquired them) in a directory that is available from your `LD_LIBRARY_PATH` (Linux), `DYLD_LIBRARY_PATH` (OSX) or, `PATH` (windows). Those binary files are available from anyone who installed MKL and have the (free) license to redistribute it with a project.

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

**Note for OSX users:** MKL installation on my OSX 10.11 placed `libiomp5.dylib` in a different folder than the rest of the MKL libraries. In such case, copy that file where it is visible, or don't rely on the MKL installation, but select the needed library files and put them somewhere on the `DYLD_LIBRARY_PATH`. In newer versions of OSX, you'd have to configure the "system integrity protection (SIP)" settings for `DYLD_LIBRARY_PATH` to be respected by the system [see more here](https://github.com/uncomplicate/neanderthal/issues/31). If you want a quick & dirty solution without much fuss, copying the `dylib` files and pasting them into `/usr/local/lib` has been reported to work by multiple users.

**Note for Windows users:** MKL installation on my Windows 10 keeps all required `.dll` files in the `<install dir>\redist` folder. The usual folders that keep `.so` and `dylib` on Linux and OSX, keep `.lib` files on Windows - you do not need those. Add the folder that contains the `dll`s into the `PATH` environment variable, and you're ready to go. Some Windows users reported that `libiomp5.dll` too is in another folder; see the note for OSX users and take the equivalent Windows action.

*Final note* If you prefer zero-install, just include `[org.bytedeco/mkl-platform-redist "2020.3-1.5.4"]` as a dependencly in your leiningen project and none of these is necessary.

## Where to Go Next

Hopefully this guide got you started and now you'd like to learn more. I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [Neanderthal API](/codox) for specific details, and feel free to take a glance at [the source](https://github.com/uncomplicate/neanderthal) while you are there.

It is also a good idea to follow [my blog at dragan.rocks](https://dragan.rocks) since I'll write about Neanderthal there.
