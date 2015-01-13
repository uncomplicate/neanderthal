(ns uncomplicate.neanderthal.examples.guides.tutorial-test
  (:require [midje.sweet :refer :all]
            [uncomplicate.neanderthal
             [core :refer :all]
             [real :refer :all]])
  (:import [java.nio ByteBuffer ByteOrder]))

"
# Neanderthal Tutorial

## Vector and Matrix data structures

Before we do any numeric computations, we have to create the data to run the
computations on.

Neanderthal uses Vectors and Matrices backed by direct byte buffers,
that can be sent to ATLAS via JNI without copying overhead.

### Creating Vectors and Matrices

Functions for creating the appropriate primitive vector or matrix are
in uncomplicate.neanderthal.real namespace. Complex vectors and matrix
creation will be in uncomplicate.neanderthal.complex, once it is implemented.

Creation functions follow the BLAS & LAPACK naming scheme,
d for doubles, f for floats, c for complex, ge for general dense matrix etc:

- dv creates a vector of doubles
- fv creates a vector of floats
- dge creates a matrix of doubles
etc.

Usually, these functions accept a previously allocated ByteBuffer,
that you can pre-populate with the data, and can also accept
sequences (mind the performance for large data, though).

All numbers in Neanderthal, both the data it holds and numbers that
the functions return are PRIMITIVE.
"

(facts
 "Here is how you create double vectors."
 (dv 10) => (dv (repeat 10 0))
 (dv 10.0) => (dv [10])
 (dv '(1 2 3 4 5 6)) => (dv 1 2 3 4 5 6)
 (dv (ByteBuffer/allocateDirect 80)) => (dv (repeat 10 0)))

(facts
 "And here is how you create double general dense matrices."
 (dge 2 3) => (dge 2 3 (repeat 6 0))
 (dge 3 2 [1 2 3 4 5 6]) => (dge 3 2 '(1 2 3 4 5 6))
 (dge 2 3 (ByteBuffer/allocateDirect 48)) => (dge 2 3 (repeat 6 0)))

"
### Neanderthal keeps data in direct byte buffers

It is awkward and cumbersome to work with buffers directly.
You should take care of endianess: java uses BIG_ENDIAN,
while Intel processors natively support LITTLE_ENDIAN. If you
pre-load your data in buffers, you, or the library you use, have to
take care of using the proper native endianess. Vertigo library
might help with this, and Neanderthal does not care how you
prepare the buffers if the data is prepared well.
Also take care to revert the buffer position to 0.

Matrix data is also kept in one-dimensional byte buffer, and NOT
in a object buffer or array that holds row buffers, for the performance
reasons. By default, when used in 2D matrices Neanderthal treats
a 1D buffer as a sequence of columns.

The same ByteBuffer can hold data for vectors as well as matrices.
"

(facts
 "Here is what you take care of if you opt to provide the initial data
in raw byte buffers."
 (let [empty-buf (ByteBuffer/allocateDirect 48)
       endianess-buf (.order ^ByteBuffer empty-buf (ByteOrder/nativeOrder))
       filled-buf (loop [i 0 buf endianess-buf]
                    (if (< i 6)
                      (recur (inc i) (.putDouble ^ByteBuffer buf (double i)))
                      buf))
       rewind-buf (.position ^ByteBuffer filled-buf 0)]
   (dv rewind-buf) => (dv (range 6))
   (dge 2 3 rewind-buf) => (dge 2 3 (range 6))
   (dge 2 3 rewind-buf) =not=> (dge 3 2 rewind-buf)))

"
Many BLAS functions works in-place! It means that they mutate the data
they work on. That way, they are faster and use less memory. These functions
have a BANG (!) suffix. They have non-destructive variants.
Keep in mind that both variants are useful for specific tasks. Usually,
we would use the destructive variants for fast algorithm internals.
"

"
## BLAS Level 1 functions

BLAS Level 1 contains functions that compute in linear time.
This means they work with 1D Vectors. Some of these functions
compute a number based on vector entries, some transform a vector
into another vector, and some

I will show you the most popular ones, so you can find
your way with others around in Neanderthal API docs.
"

(facts
 "BLAS 1 asum: Sum of absolute values."
 (asum (dv 1 2 3)) => 6.0)

(facts
 "BLAS 1 dot: Dot product is a sum of the scalar products
of respective entries of two vectors."
 (dot (dv 1 2 3) (dv 1 3 5)) => 22.0)

(facts
 "BLAS 1 copy: Here is how we copy the data from one
vector to another. We might provide the destionation and change it,
or copy the data into a nev vector."
 (let [x (dv 1 2 3)
       y (dv 3)]
   (copy! x y) => (dv 1 2 3)
   y => (dv 1 2 3)
   (copy x) => (dv 1 2 3)))

(facts
 "BLAS 1 swap: Here is how we destructively swap the entries
of two vectors."
 (let [x (dv 1 2 3)
       y (dv 1 3 5)]
   (swp! x y) => x
   x => (dv 1 3 5)
   y => (dv 1 2 3)))

(facts
 "BLAS 1 scal: Here is how you scale a vector (multiply
all entries in a vector by a scalar value)."
 (let [x (dv 1 2 3 4 5)]
   (scal! 3.0 x) => (dv 3 6 9 12 15)
   x => (dv 3 6 9 12 15)))

(facts
 "BLAS 1 axpy: SAXPY stands for Scalar a times x plus y.
It scales a vector and adds it to another vector. It can help
in acomplishing both the scaling of vectors, additions, or both.
It also have destructive and non-destructive variants, and accepts varargs.
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
   (axpy 1.5 x 2.5 y) => (dv 4 10.5 17)
   (axpy x 2.5 y (dv 1 2 2) 1.5 (dv 2 1 3)) => (dv 7.5 13 22)
   (axpy! y 1.5 x) => (dv 2.5 6 9.5)
   y => (dv 2.5 6 9.5)
   (axpy! y 2.5 x (dv 1 2 1) 1.4 (dv 2 3 4)) => (dv 8.8 17.2 23.6)))

"
## BLAS Level 2 functions

BLAS Level 2 functions are those that compute in quadratic time.
Usually, these functions combine matrices and vectors.
"

(facts
 "BLAS 2 mv: Here is how to perform a matrix-vector multiplication.
mv! is a destructive version, while mv always returns the result
in a new vector instance."
 (let [m (dge 2 3 (range 6))
       v (dv 1 2 3)
       y (dv 1 2)]
   (mv m v) => (dv 16 22)
   (mv m (dv 1 2)) => (throws IllegalArgumentException)
   (mv 1.5 m v) => (dv 24 33)
   m => (dge 2 3 (range 6))
   v => (dv 1 2 3)
   y => (dv 1 2)
   (mv! y 1.5 m v 2.5) => (dv 26.5 38.0)
   y => (dv 26.5 38.0)))

(facts
 "BLAS 2 rank: Here is how to multiply a transposed vector
by another vector, thus creating a matrix.
rank! puts the result in a provided matrix, while rank puts it
in a new matrix instance."
 (let [a (dge 2 3 (range 6))
       x (dv 1 2)
       y (dv 1 2 3)]
   (rank x y) => (dge 2 3 [1 2 2 4 3 6])
   (rank 1.5 x y) => (dge 2 3 [1.5 3.0 3.0 6.0 4.5 9.0])
   a => (dge 2 3 (range 6))
   (rank! a 2.5 x y) => (dge 2 3 [2.5 6.0 7.0 13.0 11.5 20.0])
   a => (dge 2 3 [2.5 6.0 7.0 13.0 11.5 20.0])))

"
## BLAS Level 3 functions

BLAS Level 3 functions are those that compute in cubic time.
They usually work with matrices and produce matrices.
"

(facts
 "BLAS 3 mm: Here is how you can multiply matrices. Note that
this is matrix multiplication, NOT an element-by-element multiplication,
which is a much simpler and less useful operation."
 (let [a (dge 2 3 (range 6))
       b (dge 3 1 (range 3))
       c (dge 2 1 [1 2])]
   (mm a b) => (dge 2 1 [10 13])
   (mm 1.5 a b) => (dge 2 1 [15 19.5])
   a => (dge 2 3 (range 6))
   b => (dge 3 1 (range 3))
   (mm! c a b) => (dge 2 1 [11 15])
   c => (dge 2 1 [11 15])
   (mm! c 1.5 a b 2.0) => (dge 2 1 [37 49.5])
   c => (dge 2 1 [37 49.5])))

(facts
 "Some of BLAS 1 functions work with matrices, such as copy! and swp!"
 (let [a (dge 2 3 (range 6))
       b (dge 2 3)]
   (swp! a b) => a
   a => (dge 2 3)
   b => (dge 2 3 (range 6))
   (copy! a b) => (dge 2 3)
   b => (dge 2 3)))

"
## Useful Non-BLAS functions

While BLAS functions are the meat of linear algebra computations,
there a bunch of other stuff that we would like to do with vectors
and matrices.
For example, we would like to se theis structure, dimensions,
to see specific entries, to get subvectors or submatrices,
to transpose matrices, etc.

"

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

"
Neanderthal does all these things, and does them very fast, and
usually without memory copying. You have to be careful, though.
Most of the time, when you extract a part of a matrix or a vector,
you get a live connection to the original data. All changes that
occur to the part, will also change the original.
"

(facts
 "If you change the subpart, you change the original data. It is often
useful, but if you want to avoid that, either avoid the destructive
BANG functions, or copy the data to a fresh instance before using
the BANG functions."
 (let [x (dv (range 10 20))]
   (scal! 10.0 (copy (subvector x 2 3))) => (dv 120 130 140)
   x => (dv (range 10 20))
   (scal! 10.0 (subvector x 2 3)) => (dv 120 130 140)
   x => (dv [10 11 120 130 140 15 16 17 18 19]))))

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

"
## FAST mapping and reducing

BLAS routines are fast and neat. However, often we need to compute the entries
of our matrices and vectors in a custom way. In Clojure, we would do that with
map, reduce, filter and similar functions that work the sequence.
However, sequences box all numbers, and are thus orders of magnitude slower than
functions working on primitive arrays. On the other hand, primitive arrays
areduce and amap are macros and a bit awkward...

Fortunataly, Neaanderthal comes with its own map and reduce functions that:
- Work on primitives
- Accept primitive hinted functions
- Can transform the data

This way, we get the full elegance of map and reduce with the speed (almost) as
fast as looping on primitive arrays with primitive functions. See the benchmarks
for performance details, here we only demonstrate how these are used.
"


(let [primitive-inc (fn ^double [^double x] (inc x))
      primitive-add (fn ^double [^double x ^double y ^double z] (+ x y z))
      primitive-multiply (fn ^double [^double x ^double y] (* x y))
      a (dge 2 3 (range 6))
      b (dge 2 3 (range 0 60 10))
      c (dge 2 3 (range 0 600 100))]
  (fact
   "You can change individual entries of any structure with fmap!.
You can also accumulate values with freduce, or fold the entries."
   (fmap! primitive-inc a) => (dge 2 3 (range 1 7))
   a => (dge 2 3 (range 1 7))
   (fmap! primitive-add a b c) => (dge 2 3 [1 112 223 334 445 556])
   a => (dge 2 3 [1 112 223 334 445 556])
   (freduce primitive-add 0.0 b c) => 1650.0
   (fold c) => 1500.0
   (fold primitive-multiply 1.0 a) => 2.06397368128E12))

"
## LAPACK functions

LAPACK functions build on BLAS, they are performed on the same vector
and matrix objects: dv, dge etc.

LAPACK is surrently on the TODO list. They will be added to this tutorial
once they are implemented.
"
