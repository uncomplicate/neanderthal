---
title: Fast, Native Speed, Vector Computations in Clojure
Author: Dragan Djuric
layout: article
---

For this code to work, you first need to install Neanderthal, as explained in the
Getting Started Guide. This code is also part of the test suite ([working midje tests on github](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_native_test.clj)).

## Vector and Matrix Data Structures

Before we do any numeric computations, we have to create the data to run the
computations on.

Neanderthal supports any pluggable infrastructure ([GPU computation is already
available!](tutorial_opencl.html)), and the default is to use vectors
and matrices backed by direct byte buffers that can be sent to native libraries
via JNI without copying overhead.

### Creating Vectors and Matrices

Functions for creating the appropriate primitive vectors or matrices are
in the `uncomplicate.neanderthal.native` namespace. Additional implementations
are available in appropriate namespaces; for example,
`uncomplicate.neanderthal.opencl` activates a GPU accelerated
implementation.

Import the appropriate namespaces: `core` for computation functions,
and `native` for constructors.

```Clojure

(ns uncomplicate.neanderthal.examples.guides.tutorial-native-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.neanderthal
             [core :refer :all]
             [native :refer :all]])
  (:import [java.nio ByteBuffer ByteOrder]))

```

Creation functions follow the BLAS naming scheme:
d for doubles, s for floats, c for complex, ge for general dense matrix etc:

- `dv` creates a vector of doubles
- `sv` creates a vector of floats
- `dge` creates a matrix of doubles
etc.

This tutorial will work with double-precision floats. Single-precision floats
are used in exactly the same way, except for the constructors. Single precision
does not matter much on the CPU (other that the performance of BLAS computations
may be up to 2x faster and they use 2x less storage space) but on the GPU it is
the preferred format, since current consumer-grade GPUs usually offer pro-grade
performance in single precision while being crippled for double precision 8x or
more.

All numbers in Neanderthal, both the data it holds and numbers that
the functions return are **primitive** where it matters (more about that later).

Here are a few examples written as Midje tests:

```Clojure

(facts
 "We create a few double vectors using different input methods."
 (dv 10) => (dv (repeat 10 0))
 (dv 10.0) => (dv [10])
 (dv '(1 2 3 4 5 6)) => (dv 1 2 3 4 5 6)
 (dv (ByteBuffer/allocateDirect 80)) => (dv (repeat 10 0)))

(facts
 "And here is how you create double general dense matrices."
 (dge 2 3) => (dge 2 3 (repeat 6 0))
 (dge 3 2 [1 2 3 4 5 6]) => (dge 3 2 '(1 2 3 4 5 6))
 (dge 2 3 (ByteBuffer/allocateDirect 48)) => (dge 2 3 (repeat 6 0)))

```

### Neanderthal Keeps Data in Direct Byte Buffers

Usually, these functions accept a previously allocated `ByteBuffer`, optionally
populated with data. Please note that non-buffer input source (numbers, varargs,
sequences) is suitable only as a convenience for smallish data and test code.
Be careful about the performance when working with large data, though
- sequences are slow and contain boxed numbers!

It is awkward and cumbersome to work with buffers directly. You should watch out
for endianess: Java uses BIG_ENDIAN, while Intel processors and most native
platforms natively support LITTLE_ENDIAN. If you pre-load your data in buffers,
you, or the library you use, have to take care of using the proper native
endianess. Also take care to revert the buffer position to 0. Vertigo library
might help with this, and Neanderthal does not care how you prepare the buffers
as long data is prepared well. You can use some of the existing libraries that
work with native buffers (Vertigo, etc.); check out the Neanderthal API to see what
utilities are currently available, or roll your own.

Matrix data is also kept in one-dimensional byte buffer, NOT in a object
buffer or array that holds raw buffers, for performance reasons. By default,
when used in 2D matrices, Neanderthal treats a 1D buffer as a sequence of columns.
Column-oriented order is commonly used in numerical software, contrary to
row-oriented order used by the C language. Java uses neither; 2D arrays are
arrays of array references, and this difference has a huge performance impact.
Neanderthal abstracts all these performance optimizations away, and you do not
need to care about this, unless you write a pluggable Neanderthal implementation.

The same ByteBuffer can hold data for vectors as well as matrices.

```Clojure

(facts
 "Here is what you need to take care of if you opt to provide the initial data
in raw byte buffers."
 (let [entry-width Double/BYTES
       m 2
       n 3
       empty-buf (ByteBuffer/allocateDirect (* m n entry-width))
       endianess-buf (.order ^ByteBuffer empty-buf (ByteOrder/nativeOrder))
       filled-buf (loop [i 0 buf endianess-buf]
                    (if (< i (* m n))
                      (recur (inc i) (.putDouble ^ByteBuffer buf (double i)))
                      buf))
       rewind-buf (.position ^ByteBuffer filled-buf 0)]
   (dv rewind-buf) => (dv (range (* m n)))
   (dge 2 3 rewind-buf) => (dge 2 3 (range (* m n)))
   (dge 2 3 rewind-buf) =not=> (dge 3 2 rewind-buf)))

```

## Pure and Non-pure Functions

Many BLAS functions work in-place! It means that they mutate the data they work
on. That way, they are orders of magnitude faster and use less memory. These
functions have a BANG (`!`) suffix. They have non-destructive, pure variants,
without the `!`. Keep in mind that both variants are useful for specific tasks.
Usually, we should use the destructive variants for fast algorithm internals or
with huge data when there is no space for copying.

## BLAS Level 1 Functions

BLAS Level 1 contains functions that compute in linear time, O(n). They usually
work with 1D vectors, but some of them are also appropriate for 2D matrices.
Some of these functions compute a number based on vector entries while some
transform the values of entries.

I will show you the most popular ones, so you can easily find your way with
others in Neanderthal API docs.

```Clojure

(facts
 "BLAS 1 asum: Sum of absolute values."
 (asum (dv 1 2 -5)) => 8.0)

(facts
 "BLAS 1 sum: Sum of all values. Note: this function is not a part of
BLAS standard."
 (sum (dv 1 2 -5)) => -2.0)

(facts
 "BLAS 1 dot: Dot product is a sum of the scalar products of respective entries
of two vectors."
 (dot (dv 1 2 3) (dv 1 3 5)) => 22.0)

(facts
 "BLAS 1 copy: Here is how we copy the data from one vector to another. We may
provide the destionation and change it, or copy the data into a new vector.
And it works with matrices, too."
 (let [x (dv 1 2 3)
       y (dv 3)
       a (dge 2 3 (range 6))
       b (dge 2 3)]
   (copy! x y) => (dv 1 2 3)
   y => (dv 1 2 3)
   (copy x) => (dv 1 2 3)
   (copy! a b) => (dge 2 3 (range 6))
   b => (dge 2 3 (range 6))))

(facts
 "BLAS 1 swap: Here is how we destructively swap the entries of two vectors or
matrices."
 (let [x (dv 1 2 3)
       y (dv 1 3 5)
       a (dge 2 3 (range 6))
       b (dge 2 3)]
   (swp! x y) => x
   x => (dv 1 3 5)
   y => (dv 1 2 3)
   (swp! a b) => a
   a => (dge 2 3)
   b => (dge 2 3 (range 6))))

(facts
 "BLAS 1 scal: Here is how you scale a vector (multiply all entries in a vector
by a scalar value). Also works on matrices."
 (let [x (dv 1 2 3 4 5)]
   (scal! 3.0 x) => (dv 3 6 9 12 15)
   x => (dv 3 6 9 12 15)))

(facts
 "BLAS 1 axpy: SAXPY stands for Scalar a times x plus y. It scales a vector and
adds it to another vector. It can help in acomplishing both the scaling of
vectors, additions, or both. It also works on matrices.
It has destructive and non-destructive variants and accepts varargs:
ax - scaling
xpy - vector addition
axpy - scaling and addition
axpy! - destructive scaling and addition
"
 (let [x (dv 1 2 3)
       y (dv 1 3 5)]
   (ax 2.5 x) => (dv 2.5 5.0 7.5)
   x => (dv 1 2 3)
   (xpy x y) => (dv 2 5 8)
   (xpy x y (dv 3 2 1) x) => (dv 6 9 12)
   x => (dv 1 2 3)
   (axpy x y) => (xpy x y)
   (axpy x y (dv 3 2 1) x) => (xpy x y (dv 3 2 1) x)
   (axpy 1.5 x (zero x) 2.5 y) => (dv 4 10.5 17)
   (axpy 2.5 y x (dv 1 2 2) 1.5 (dv 2 1 3)) => (dv 7.5 13 22)
   (axpy! 1.5 x y) => (dv 2.5 6 9.5)
   y => (dv 2.5 6 9.5)
   (axpy! 2.5 x y (dv 1 2 1) 1.4 (dv 2 3 4)) => (dv 8.8 17.2 23.6)))

```

## BLAS Level 2 Functions

BLAS Level 2 functions are those that compute in quadratic time O(n^2).
Usually, these functions combine matrices and vectors.

```Clojure

(facts
 "BLAS 2 mv: Here is how to perform a matrix-vector multiplication.
mv! is a destructive version, while mv always returns the result in a new vector
instance."
 (let [m (dge 2 3 (range 6))
       v (dv 1 2 3)
       y (dv 1 2)]
   (mv m v) => (dv 16 22)
   (mv m (dv 1 2)) => (throws IllegalArgumentException)
   (mv 1.5 m v) => (dv 24 33)
   m => (dge 2 3 (range 6))
   v => (dv 1 2 3)
   y => (dv 1 2)
   (mv! 1.5 m v 2.5 y) => (dv 26.5 38.0)
   y => (dv 26.5 38.0)))

(facts
 "BLAS 2 rank: Here is how to multiply a transposed vector by another vector,
thus creating a matrix. rank! puts the result in a provided matrix, while rank
puts it in a new matrix instance."
 (let [a (dge 2 3 (range 6))
       x (dv 1 2)
       y (dv 1 2 3)]
   (rank x y) => (dge 2 3 [1 2 2 4 3 6])
   (rank 1.5 x y) => (dge 2 3 [1.5 3.0 3.0 6.0 4.5 9.0])
   a => (dge 2 3 (range 6))
   (rank! 2.5 x y a) => (dge 2 3 [2.5 6.0 7.0 13.0 11.5 20.0])
   a => (dge 2 3 [2.5 6.0 7.0 13.0 11.5 20.0])))

```

## BLAS Level 3 Functions

BLAS Level 3 functions are those that compute in cubic time O(n^3).
They usually work with matrices and produce matrices.

```Clojure

(facts
 "BLAS 3 mm: Here is how you can multiply matrices. Note that this is matrix
multiplication, NOT an element-by-element multiplication, which is a much
simpler and less useful operation."
 (let [a (dge 2 3 (range 6))
       b (dge 3 1 (range 3))
       c (dge 2 1 [1 2])]
   (mm a b) => (dge 2 1 [10 13])
   (mm 1.5 a b) => (dge 2 1 [15 19.5])
   a => (dge 2 3 (range 6))
   b => (dge 3 1 (range 3))
   (mm! a b c) => (dge 2 1 [11 15])
   c => (dge 2 1 [11 15])
   (mm! 1.5 a b 2.0 c) => (dge 2 1 [37 49.5])
   c => (dge 2 1 [37 49.5])))

(facts
 "Some of BLAS 1 functions, such as copy!, and swp!, axpy!, work with matrices."
 (let [a (dge 2 3 (range 6))
       b (dge 2 3)]
   (swp! a b) => a
   a => (dge 2 3)
   b => (dge 2 3 (range 6))
   (copy! a b) => (dge 2 3)
   b => (dge 2 3)))

```

## Useful Non-BLAS Functions

While BLAS functions are the meat of linear algebra computations, there is a bunch
of other stuff that we would like to do with vectors and matrices. For example,
we would like to see their structure, dimensions, specific entries, to get
subvectors or submatrices, to transpose matrices, etc. Neanderthal offers time
and space efficient implementations of such operations.

```Clojure

(facts
 "Miscelaneous vector functions."
 (let [x (dv (range 10 20))]
   (= x (dv (range 10 20))) => true
   (identical? x (dv (range 10 20))) => false
   (entry x 7) => 17.0
   (dim x) => 10
   (vect? x) => true
   (vector? x) => false
   (vect? [1 2 3]) => false
   (zero x) => (dv 10)
   (subvector x 2 3) => (dv 12 13 14)
   x => (dv (range 10 20)))

 (facts
  "Miscelaneous matrix functions."
  (let [a (dge 20 30 (range 600))]
    (= a (dge 20 30 (range 600))) => true
    (identical? a (dge 20 30 (range 600))) => false
    (entry a 15 16) => 335.0
    (mrows a) => 20
    (ncols a) => 30
    (trans (trans a)) => a
    (col a 2) => (dv (range 40 60))
    (row a 3) => (dv (range 3 600 20))
    (matrix? a) => true
    (matrix? (row a 4)) => false
    (zero a) => (dge 20 30)
    (submatrix a 11 12 2 3) => (dge 2 3 [251 252 271 272 291 292])
    (ax 2.0 (col a 0)) => (dv (range 0 40 2))
    a => (dge 20 30 (range 600))
    (scal! 2.0 (row a 3)) => (row a 3)
    a =not=> (dge 20 30 (range 600))))

```

Neanderthal does all these things, and does them very fast, usually without
memory copying. You have to be careful, though. Most of the time, when you
extract a part of a matrix or a vector, you get a live connection to the
original data. All changes that occur to the part will also change the original.
It is often useful for performance reasons, but sometimes you want to avoid it.
In that case, avoid the destructive BANG functions, or copy the data to a fresh
instance before using the BANG functions. The important thing is that you always
have control and can explicitly choose what you need in particular case:
purity or performance, or sometimes both.

```Clojure

(facts
 "If you change the subpart, you change the original data. "
 (let [x (dv (range 10 20))]
   (scal! 10.0 (copy (subvector x 2 3))) => (dv 120 130 140)
   x => (dv (range 10 20))
   (scal! 10.0 (subvector x 2 3)) => (dv 120 130 140)
   x => (dv [10 11 120 130 140 15 16 17 18 19]))))

```

## **Fast** Mapping and Reducing

BLAS routines are fast and neat. However, often we need to compute the entries
of our matrices and vectors in a custom way. In Clojure, we would do that with
map, reduce, filter and similar functions that work the sequence. However,
sequences box all numbers, and are thus orders of magnitude slower than
functions working on primitive arrays. On the other hand, primitive arrays
areduce and amap are macros and a bit awkward...

Fortunataly, Neanderthal comes with its own map and reduce functions that:

- Work on primitives
- Accept primitive hinted functions
- Can transform the data

This way, we get the full elegance of map and reduce with the speed (almost) as
fast as looping on primitive arrays with primitive functions. See the benchmarks
for performance details, here we only demonstrate how these are used.

```Clojure

(let [primitive-inc (fn ^double [^double x] (inc x))
      primitive-add (fn ^double [^double x ^double y ^double z] (+ x y z))
      primitive-multiply (fn ^double [^double x ^double y] (* x y))
      a (dge 2 3 (range 6))
      b (dge 2 3 (range 0 60 10))
      c (dge 2 3 (range 0 600 100))]
  (fact
   "You can change individual entries of any structure with fmap!. You can also
accumulate values with freduce, or fold the entries."
   (fmap! primitive-inc a) => (dge 2 3 (range 1 7))
   a => (dge 2 3 (range 1 7))
   (fmap! primitive-add a b c) => (dge 2 3 [1 112 223 334 445 556])
   a => (dge 2 3 [1 112 223 334 445 556])
   (freduce primitive-add 0.0 b c) => 1650.0
   (fold c) => 1500.0
   (fold primitive-multiply 1.0 a) => 2.06397368128E12))

```

## LAPACK Functions

LAPACK functions build on BLAS, they are performed on the same vector and matrix
objects: dv, dge etc.

LAPACK is currently on the TODO list. It will be added to this tutorial
once these functions are implemented.

## Can Neanderthal Go Even Faster?

Yes, it can, and MUCH. Neanderthal have a pluggable infrastructure, and
already comes with a GPU engine that can process your data on graphic cards
orders of magnitude faster than on the CPU. Check out the [GPU tutorial](tutorial_opencl.html)!

## Additional Examples

Neanderthal comes with additional examples from the Linear algebra literature.
Check out the [Examples](https://github.com/uncomplicate/neanderthal/tree/master/test/uncomplicate/neanderthal/examples)
folder for the always accurate and working examples in the form of midje tests.
