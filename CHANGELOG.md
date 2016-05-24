# [Neanderthal](http://neanderthal.uncomplicate.org) - notable changes between versions

## 0.6.2

* Updated ClojureCL dependency to 0.6.4

## 0.6.1

* Updated ClojureCL dependency to 0.6.3

## 0.6.0

New features

* Completely new OpenCL engine for GPU matrix computing - **supports AMD, Nvidia, and Intel, on Linux, Windows, and OSX
* Support Fluokitten's Monoid and Magma in vectors and matrices
* transfer method in core that always transfers data to host memory

Changes:

* opencl methods renamed
* default OpenCL engine changed to clblast
* old amd-gcn engine deprecated

## 0.5.0

New features:
* Streamlined dependencies: no longer need 2 dependencies in project files. The dependency on uncomplicate/neanderthal is enough
* Comes with Mac OS X build out of the box. No need even for external ATLAS.
* release and with-release moved from ClojureCL to uncomplicate/commons
* Support for Fluokitten's fmap!, fmap, fold, foldmap, op...

## 0.4.0

New features:

* Streamlined factory-based constructors in core.
* OpenCL vectors and matrices now support equality comparisons, offsets, strides,
subvectors, and submatrices. Matrices now can be swapped and copied.

Changes:

* OpenCL read! and write! replaced with generic transfer! multimethod that supports
a much wider area of memory types to move data to and from.
* A large number of internal implementation changes that should not affect end users
(other than as removing bugs).
* Several important bugfixes (see git commit history).

## 0.3.0

New features:

* Support for pluggable BLAS engines
* GPU computing engine based on OpenCL (kernels optimized for AMD for now)

Changes:

* Reorganized namespaces - now almost complete public API is in the core namespace
* Changed the order of parameters in axpy!, mv! and mm! (and their variants)

## 0.2.0

New features:

* implemented BLAS support for floats
* implemented fmap!, freduce, and fold functions for all existing types of matrices and vectors

Changes:

No API changes were required for these features.
