# [Neanderthal](http://neanderthal.uncomplicate.org) - notable changes between versions

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
