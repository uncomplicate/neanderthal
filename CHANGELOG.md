# [Neanderthal](http://neanderthal.uncomplicate.org) - notable changes between versions

## 0.15.0

### Fixes

* Fix #29 - OpenCL engine does not try to load CUDA-related stuff any more.

## 0.14.0

### Breaking changes:

* tr* work with LUFactorization instead of GEMatrix.

### New features:

* Matrix inverse.
* Condition number.
* Pure tr* methods.

### Enhancements

* Improved TRMatrix printing.

## 0.13.0

* Support for chained matrix multiplication in mm.

## 0.12.0

* Support for inverting matrices through trf/tri.

## 0.11.0

### New features:

* CUDA/cuBLAS based engine (requires CUDA toolkit).
* Additional methods from Blas supported by matrices.

## 0.10.0

### New features:

* Added aux namespace for auxiliary functions.
* Sorting of vectors, GE, and TR host matrices.
* Bulk alter! method added.
* view-vctr and view-ge support stride multiplier.

### Enhancements

* set-all accepts NaN
* CL factories implement FactoryProvider.

## 0.9.0

### New features:

* Linear algebra functions (LAPACK).
* Support for TR matrices.
* Pretty-printing
* GE and TR support some more BLAS-1 functions.

### Enhancements

* Cheat Sheet in the docs.
* Updated JOCLBlast dependency to 0.10.0.
* Updated Fluokitten dependency to 0.6.0.
* Internal api and implementations made more straightforward.

### Breaking changes:

* Naming scheme changed from single to float for single-precision structures.
* sv and sge from the native namespace renamed to fv and fge.
* core constructor functions changed from create to vctr, ge, tr, etc.
* Core constructors no longer accept raw buffers.

## 0.8.0

### Changes

* Removed the amd-gcn engine. Use clblast engine instead.

### Enhancements

* The native part is now compiled for Linux, MacOX AND Windows
* one-argument pow function added.
* native-factory method added to FactoryProvider protocol.
* factories and data accessors implement the compatible method.
* Updated JOCLBlast dependency to 0.9.0.

### Bugfixes

* Attempting (create -m -n) now throws IllegalArgumentException
* Fixed a sum bug in native implementation when stride is not 1.

## 0.7.0

### New features:

* scal implementation for matrices.

### Bugfixes:

* Fixed a Buffer.limit bug in subvector and submatrix.
* Fixes #15

### Enhancements

* native function in core
* one-argument fold now use sum instead of looping.
* Updated JOCLBlast dependency to 0.8.0 (also fixes #15)

## 0.6.2

* Updated ClojureCL dependency to 0.6.4

## 0.6.1

* Updated ClojureCL dependency to 0.6.3

## 0.6.0

### New features

* Completely new OpenCL engine for GPU matrix computing - **supports AMD, Nvidia, and Intel, on Linux, Windows, and OSX
* Support Fluokitten's Monoid and Magma in vectors and matrices
* transfer method in core that always transfers data to host memory

### Changes

* opencl methods renamed
* default OpenCL engine changed to clblast
* old amd-gcn engine deprecated

## 0.5.0

### New features

* Streamlined dependencies: no longer need 2 dependencies in project files. The dependency on uncomplicate/neanderthal is enough
* Comes with Mac OS X build out of the box. No need even for external ATLAS.
* release and with-release moved from ClojureCL to uncomplicate/commons
* Support for Fluokitten's fmap!, fmap, fold, foldmap, op...

## 0.4.0

### New features

* Streamlined factory-based constructors in core.
* OpenCL vectors and matrices now support equality comparisons, offsets, strides,
subvectors, and submatrices. Matrices now can be swapped and copied.

### Changes

* OpenCL read! and write! replaced with generic transfer! multimethod that supports
a much wider area of memory types to move data to and from.
* A large number of internal implementation changes that should not affect end users
(other than as removing bugs).
* Several important bugfixes (see git commit history).

## 0.3.0

### New features

* Support for pluggable BLAS engines
* GPU computing engine based on OpenCL (kernels optimized for AMD for now)

### Changes

* Reorganized namespaces - now almost complete public API is in the core namespace
* Changed the order of parameters in axpy!, mv! and mm! (and their variants)

## 0.2.0

### New features

* implemented BLAS support for floats
* implemented fmap!, freduce, and fold functions for all existing types of matrices and vectors

### Changes

No API changes were required for these features.
