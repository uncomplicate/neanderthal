---
title: "Get Started"
Author: Dragan Djuric
layout: article
---

Neanderthal's default option is to use use native libraries, so it is very important that you do not skip any part of this guide.

## How to Get Started

* Walk through this guide, set up your development environment, and try the examples.
* Familiarize yourself with Neanderthal's [more detailed tutorials](/articles/guides.html) and [API documentation](/codox).

## Installation (Leiningen)

The most straightforward way to include Neanderthal in your project is with Leiningen. **Check [the Hello World project](https://github.com/uncomplicate/neanderthal/blob/master/examples/hello-world-aot/project.clj) out for the complete example.** Please note that, if you need an older version of Neanderthal, they may need a bit more specific installation steps, which are explained in [Requirements](#requirements).

* Add the following dependency to your `project.clj`,: ![](https://clojars.org/uncomplicate/neanderthal/latest-version.svg)
* Add Intel MKL distribution jar `[org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]` as your project's dependency (Linux or Windows). for MacOS, the native binaries are already there on the OS, so you don't need MKL (nor MKL works on Mac).
* (optional) Add a CUDA distribution jar as your project's dependency (`[org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.143830-1" :classifier "linux-x86_64-redist"]` on Linux, or `[org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.145546-3" :classifier "windows-x86_64-redist"]` on Windows). Please note that you'll have to add Sonatype snapshots Maven repository (`:repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]` since Maven Central introduced hard limit of 2GB, which is exceeded by CUDA's 3GB distribution). MacOS doesn't ship with Nvidia GPUs, thus doesn't support CUDA.

Neanderhtal will use the native CPU MKL (and/or CUDA, OpenCL, OpenBLAS, or Accelerate) binaries from these jars automatically, so you don't need to do anything else. If the jars are not present, Neanderthal will expect you to have a system-wide MKL (OpenBLAS, CUDA, etc.) installation as explained in [Native Engine Requirements](#legacy-0470-and-older-the-native-library-used-by-neanderthals-native-engine-optional). **Note: MKL distribution size is 200 MB, while CUDA is 2.9 GB!** Leiningen will download these JARs the first time you include them, which might take some time, so it's a good idea to run `lein deps` *in the terminal* and wait each time you update the version. If you let your IDE do this, it will quietly download under the hood without visibly reporting this to you, which might confuse you into thinking something's not right while you wait for 20 minutes for your REPL to open. Even worse, if you kill that process, you might end with a broken jar in your local Maven repository.

Also note the `:classifier`, which is OS dependent. You may omit it, but then binaries for all operating systems will be downloaded, which will take longer. For, Windows, use `windows-x86_64-redist`. For MacOS, you do not need this, as it ships with Accelerate framework.

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

## Overview and Features

Neanderthal is a Clojure library for fast matrix and linear algebra computations that supports pluggable engines:

* Currently there are 3 **native engines**! The first is for Intel CPUs, based on a highly optimized native [Intel's oneMKL](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) library of [BLAS](https://netlib.org/blas/) and [LAPACK](https://www.netlib.org/lapack/) computation routines (MKL is not open-source, but it is free to use and redistribute since 2016). The second one is for Mac, and it is based on Apple's accelerate. You might also use [OpenBLAS](https://github.com/OpenMathLib/OpenBLAS), which supports both Intel and Apple, and all 3 operating systems, but does not support all Neanderthal features.
* The **CUDA GPU engine** is based on cuBLAS and supports all modern Nvidia GPUs. It uses [ClojureCUDA](https://clojurecuda.uncomplicate.org).. Check out [Uncomplicate ClojureCUDA](https://clojurecuda.uncomplicate.org) for more info.
* The **OpenCL GPU engine** is based on OpenCL BLAS routines from [CLBlast](https://github.com/CNugteren/CLBlast) library for even more computational power when needed. It uses [ClojureCL](https://clojurecl.uncomplicate.org) and [JOCL](https://jocl.org) libraries. Check out [Uncomplicate ClojureCL](https://clojurecl.uncomplicate.org).

### Implemented Features

* Data structures: double and single precision vectors, dense matrices (GE), triangular matrices (TR), symmetric matrices (SY), banded, diagonal, etc.;
* BLAS Level 1, 2, and 3 routines;
* Lots of (but not all) LAPACK routines;
* Sparse matrices and vectors.
* Various Clojure vector and matrix functions (transpositions, submatrices etc.);
* Easy and efficient data mapping and transfer to and from GPUs;
* Fast map, reduce and fold implementations for the provided structures.
* OpenCL GPU support
* CUDA GPU support
* Intel (Linux and Windows) and Apple Silicon processors.

### On the TODO List

* ~"Tensors"~: This is already available in [Deep Diamond](https://github.com/uncomplicate/deep-diamond)!
* Support for complex numbers;

## Requirements

You need at least Java 8, but the newer the better.

### JVM requirements

Depending on the OpenJDK version, you may or may not be required to set these options, but I recommend that you set them anyway (the first one will ensure there are no reflections in Neanderthal's macro-heavy code):

```clojure
:jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                     "--enable-native-access=ALL-UNNAMED"]}
```

### CPU engine Binaries

Neanderthal uses MKL, Accelerate, and/or OpenBLAS. Although Neanderthal ships with the code to access routines from these libraries, the binaries themselves are available in two ways:
* You can include them through Maven redist jars (see the Hello World project).
* You may install them globally on your system through your system's package manager. In this case, you don't need the redist jars, but you must ensure that the version that you have on your OS is compatible with what Neanderthal uses.

### GPU drivers for the OpenCL GPU engine

Everything will magically work (no need to compile anything) on Nvidia, AMD, and Intel's GPUs and CPUs as long as you have appropriate GPU drivers.

Works on Linux, and Windows; sadly, deprecated on MacOS!

Follow the [ClojureCL getting started guide](https://clojurecl.uncomplicate.org/articles/getting_started.html) for the links for the GPU platform that you use and more detailed info.

**If you use a pre-2.0 OpenCL platform (Nvidia), you'll have to use `command-queue-1` and/or `with-default-1` instead of `command-queue` and `with-default`.**

### GPU drivers for the CUDA GPU engine

Everything will magically work (no need to compile anything) on Nvidia, provided that you **have Nvidia drivers**, and included the appropriate
Nvidia Toolkit JavaCPP jar (`[org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.143830-1" :classifier "linux-x86_64-redist"]` or `[org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.145546-3" :classifier "windows-x86_64-redist"]`).
Similarly to the CPU engine binaries, you can either include the redist jars in your project, or you can install CUDA toolkit globally through your operating system package manager.

Follow the [ClojureCUDA getting started guide](https://clojurecuda.uncomplicate.org/articles/getting_started.html) for the links for the GPU platform that you use and more detailed info.

*macOS* doesn't support CUDA 11 and higher (and Apple hasn't shipped Nvidia GPUs since 2014 anyway).

## Where to Go Next

Hopefully this guide got you started and now you'd like to learn more. I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [Neanderthal API](/codox) for specific details, and feel free to take a glance at [the source](https://github.com/uncomplicate/neanderthal) while you are there.

It is also a good idea to follow [my blog at dragan.rocks](https://dragan.rocks) since I'll write about Neanderthal there.

Also check out the [BOOKS](http://aiprobook.com). They not only demonstrate in great detail how to effectively and efficiently use Neanderthal, but they also help me fund this work!
