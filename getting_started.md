---
title: "Get Started"
Author: Dragan Djuric
layout: article
---

Neanderthal uses native libraries, so it is very important that you do not skip any part of this guide.

# How to get started
* Walk through this guide, set up your development environment, and try the examples.
* Familiarize yourself with Neanderthal's [more detailed tutorials](/articles/guides.html) and [API documentation](/codox).

# Overview and features

Neanderthal is a Clojure library for fast matrix and linear algebra computations based on the highly optimized [Automatically Tuned Linear Algebra Software (ATLAS)](http://math-atlas.sourceforge.net/) native library of [BLAS](http://netlib.org/blas/) and [LAPACK](http://www.netlib.org/lapack/) computation routines. It provides the following features:

## Implemented features

* Data structures: double vector, double general dense matrix (GE);
* BLAS Level 1, 2, and 3 routines;
* Various Clojure vector and matrix functions (transpositions, submatrices etc.);
* Fast map, reduce and fold implementations for the provided structures.

## On the TODO list

* LAPACK routines;
* Banded, symmetric, triangular, and sparse matrices;
* Support for complex numbers;
* Support for single-precision floats.

# Installation

1. Install ATLAS on your system following the [ATLAS Installation Guide](http://math-atlas.sourceforge.net/atlas_install/atlas_install.html). (see [Requirements](#requirements))
2. Add Neanderthal jars to your classpath.

## With Leiningen

The most straightforward way to include Neanderthal in your project is with Leiningen. Add the following dependencies to your `project.clj`, just like in [the Hello World project](https://github.com/uncomplicate/neanderthal/blob/master/examples/hello-world/project.clj):

```clojure
[uncomplicate/neanderthal "0.1.0"]
[uncomplicate/neanderthal-atlas "0.1.0" :classifier "amd64-Linux-gpp-jni"]
```

Replace `amd64-Linux-gpp-jni` with your system's architecture and OS classifier. If you are not sure what exactly to write, check `(System/getProperty "os.arch")` and `(System/getProperty "os.name")`.

I will always provide at least the Linux library in Clojars. If the library for your OS is not in Clojars, checkout [neanderthal-atlas](https://github.com/uncomplicate/neanderthal-atlas) source and build it with maven using `mvn install`. The build is fully automatic if you have gcc and other related gnu tools. If you can successfully build ATLAS, you already have all the necessary tools.

## Requirements

Neanderthal is a Clojure library packaged in two `jar` files, distributed through [Clojars](http://clojars.org). One is a pure Clojure library that you will use directly, and the other contains native JNI bindings for a specific operating system. They follow [maven](http://www.maven.org) (and [leiningen](http://www.leiningen.org)) naming conventions:

* Pure Clojure lib: `groupId` is `uncomplicate` and `artifactId` is `neanderthal`.
* Native JNI lib: `groupId` is `uncomplicate` and `artifactId` is `neanderthal-atlas` with a classifier for your operating system and architecture, e.g. `amd64-Linux-gpp-jni`.

Neanderthal also **uses the native ATLAS library and expects that you make it available on your system, typically as a shared so, dll, or dylib!** ATLAS is highly optimized for various architectures - if you want top performance **you have to build ATLAS from the source**. Do not worry, ATLAS comes with automatic autotools build script, and a [detailed configuration and installation guide](http://math-atlas.sourceforge.net/atlas_install/atlas_install.html). If you do not follow this procedure, and use a pre-packaged ATLAS provided by your system (if it exists), you will probably get degraded performance compared to a properly installed ATLAS.

# Usage

First `use` or `require` `uncomplicate.neanderthal.core` and `uncomplicate.neanderthal.real` in your namespace, and you'll be able to call appropriate functions from the Neanderthal library.

```clojure
(ns example
  (:use [uncomplicate.neanderthal core real]))
```

Now you can create neanderthal vectors and matrices, and do computations with them.

Here is how we create two double-precision vectors and compute their dot product:

```clojure
(def x (dv 1 2 3))
(def y (dv 10 20 30))
(dot x y)
```

This is one of the ways we can multiply matrices:

```clojure
(def a (dge 3 2 [1 2 3 4 5 6))
(def b (dge 2 3 [10 20 30 40 50 60]))
(mm a b)
```

# Where to go next

Hopefully this guide got you started and now you'd like to learn more. I expect to build a comprehensive base of articles and references for exploring this topic, so please check the [All Guides](/articles/guides.html) page from time to time. Of course, you should also check the [Neanderthal API](/codox) for specific details, and feel free to take a gander at [the source](https://github.com/uncomplicate/neanderthal) while you are there.
