# [Neanderthal](http://neanderthal.uncomplicate.org) - notable changes between versions

# 0.30.0

* Fix regression in core.rk! caused by 0.30.0 update.

# 0.30.0

* New function mmt! for symmetric matrix multiply with its transpose (faster than mm!).
* Support for symmetric rk! of a vector with its transpose.

# 0.29.0

* Support mapping a vector to a file channel.
* Support int, long, short, and byte vectors.

# 0.28.0

* Fix a subtle CUDA vect-math return object bug.

# 0.27.0

* Support CUDA 10.2.0
* A bunch of bugfixes provided by Kamil Toman (katox).

# 0.26.0

* view and view-* behavior made more consistent.

# 0.25.0

* Random number generation of vector entries with uniformly and normally distributed numbers.
* Renamed aux to auxil to work around a Windows bug of not allowing files named aux.

# 0.23.0

* Added copy-sign, ramp, step, and sigmoid functions to math and vect-math (MKL, CUDA, OpenCL)

# 0.22.0

* Introduced the Viewable/view protocol and view function (interop).
* CLBlast bumped to 1.5.0/

# 0.21.0

* Clojure upgraded to 1.10.
* Misc bugfixes.

# 0.20.0

* SDD available as a SVD implementation.
* Eigenvalues and eigenvectors computing available for symmetric matrices.
* Fluokitten performance regression (introduced in 0.18.0) fixed.
* Fluokitten support in non-double objects.
* Fluokitten accepts non-primitive function for Neanderthal objects.
* Custom non-blas sum function sped up on CPU.
* JCublas upgraded to 0.9.2.

# 0.19.0

* Support explicit stream in memcpy.
* CUDA engine uses explicit context.
* sum support in CUDA matrices.
* TRSV in OpenCL matrices.
* CLBlast dependency updated to 1.3.0. Context creation for OpenCL is much faster now.
* Vertigo dependency removed.
* view-ge supports arbitrary dimensions now.
* ge supports nested sequence as source for its rows.

## 0.18.1

* Added FlowProvider/flow to internal core.

## 0.18.0

* Updated to Java 9 modules. Requires add-open jvm argument if run with JDK 9+.
* Clojure dep updated to 1.9.0.
* Upgrades JCuda dependency to 0.9.0, supports CUDA 9.
* Core constructors accept any factory provider as factory.
* GPU objects are safe to print after the factory has been released.

## 0.17.2

* Fix the uplo_modf bug (#33).

## 0.17.1

* Upgraded JOCLBlast dependency to 1.2.0.

## 0.17.0

### New features

* Vectorized mathematical functions (cca 50 pieces in the vect-math namespace).
* New functions in the math namespace to support scalar equivalents of vect-math functions.
* Schur decomposition!

### Enhancements

* JOCLBlast engine upgraded to 1.1.0.
* CUDA implementation of SY matrices.
* OpenCL implementation of SY matrices.

## 0.16.1

### Bugfixes

* Fixed call with wrong number of arguments for the transpose of OrthogonalFactorization.

## 0.16.0

### New features

* Diagonal matrices (GD)
* Tridiagonal matrices (GT)
* Diagonally dominant tridiagonal matrices (DT)
* Symmetric tridiagonal matrices (ST)

### Enhancements

* Updated JOCLBlast dependency to 1.0.1.
* Orthogonal factorizations greatly simplified
* Symmetric and triangular mm support more layout and a/b position variations.
* Upgraded Intel MKL to 2018 (it should work with earlier versions, but YMMV).

### Breaking changes

* New simplified orthogonal factorization related functions replace the old api in linalg.

### Bugfixes

* Fixed TR and SY mm and mv when k=0.
* Fixed transpose implementations in various non-GE matrices.

## 0.15.0

### New features

* Symmetric matrices (SY)
* Banded matrices (GB, TB, SB)
* Packed matrices (TP, SP)

### Enhancements

* Better printing
* Fluokitten protocols supported by all matrix types.
* Overhaul of internals that opens easier path for new matrix types and specialized engines.
* Automatized triangular factorizations and solvers.

### Breaking changes

* :order replaced by :layout in matrix options.

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
