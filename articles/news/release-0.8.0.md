---
title: "Neanderthal 0.8.0: CPU and GPU support on  Linux, Windows, and OS X!"
Author: Dragan Djuric
layout: article
---

9.10.2016.

In this release, spotlight is on the Windows build of the native CPU BLAS/LAPACK engine!

This is something that has already been supported, but GNU buildchain made
it challenging for Windows users to do themselves. From this release on,
Neanderthal's native engine comes pre-compiled in the jar. What is left to the
users is to supply libatlas.dll system library on their path. Since that requires
building atlas (still might be challenging to some users), I'll also send libatlas.dll
optimized for my laptop upon request (please see the Getting Started page for the directions).

So, new Neanderthal:

* Works on all three major hardware platforms: AMD, Nvidia, and Intel
* Works on all three major operating systems: Linux, Windows, and OS X
* Is faster and easier to use than ever.

* Version 0.8.0 is in [clojars](https://clojars.org/uncomplicate/neanderthal)
* [GPU tutorial (updated for 0.8.0)](articles/tutorial_opencl.html)
* [Getting started guide](articles/getting_started.html)
* [CHANGELOG](https://github.com/uncomplicate/neanderthal/blob/master/CHANGELOG.md)
