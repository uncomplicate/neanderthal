---
title: Matrix Computations on the GPU in Clojure (in TFLOPS!)
Author: Dragan Djuric
layout: article
---

**First some explanations; code comes later..** ([working midje tests on github](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/examples/guides/tutorial_opencl_test.clj))

So, you've [installed Neanderthal](getting_started.html) and [learned
how to work](tutorial_native.html) with vectors and matrices on the CPU at
roughly [10x speed of pure Java libs](benchmarks.html). It is fast, running at tens of GFLOPS,
but you've read that nowadays the computations on GPUs are what all the hipster
geeks do, so you wonder whether there is something to it. If only you could
connect to [CUBLAS](https://developer.nvidia.com/cuBLAS), your algorithms would
speed up thousandfold with almost no sweat, like in the commercials...

I have news for you:

* The bad: it is not that simple. Your algorithms probably aren't performant on massively
parallel architectures, and you'd need to learn quite a few new tricks to collect
the benefits you see on NVIDIA and AMD websites.

* The good: Neanderthal implements BLAS algorithms and abstracts most of that
complexity for vector and matrix computations away behind a frendly Clojure API.

**tl;dr: Multiplication of large matrices is more than 500x faster with Neanderthal
than with optimized pure Java libraries, 25x faster than Neanderthal native engine
and some thousands times faster than the nested looping code with primitives
that you'd write yourself.**

## What You Need to Know About GPU Programming Before We Begin

The most important thing when working with parallel accellerators (GPUs and others)
is to be aware that a large part of your code will run on a physically separate
device, which also has its own working memory. It cannot access data from the
main memory, such as your objects, arrays, primitives and such. It also cannot
run your arbitrary program code (Java, C, R). Even if/when it could (Aparapi)
it sucks at it big time, because it is made of thousans of very dumb processors that
are excellent at one thing only - raw computations - and suck at everything else,
including logic.

A typical approach is to have your program consist of two parts: host and device.
The host is a typical Clojure (Java) program that do the usual things Clojure programs
do: talk to the web, format and extract data, interface with the database,
recursively compute factorials, and perform other logic-heavy tasks.
Then once it has collected the data and stuffed it in a
big raw chunk of bazillion primitive floating point numbers, it sends it to the
device memory and tells the device to run the kernel programs and compute the data.
When the main program wants to see the results, it has to transfer them from the
device memory to the main host memory.

Obviously, if the computations are not intensive and demanding, the whole
host/device communication might eat all the performance benefits. That's why you
should not bother to send two numbers to the GPU to be computed in microseconds
('thanks' to the communication overhead) when even Java would compute that in
a nanosecond; in the same way as you should not call a REST service each time
you need to add two numbers.

So, the moral of this story is: avoid copying data to and from the device unless
it is absolutely necessary. Even avoid any communication with the device unless
it is absolutely necessary.

The typical workflow woud be: prepare the input data in your Clojure program and
send it to the device, then call many Neanderthal and ClojureCL functions that
work with that data without transferring it back. Only transfer the final result.

You'll still write your algorithms in Clojure as any normal Clojure code,
you just have to be aware that the data is actually on the device, and that the
Neanderthal functions you call run the kernels (supplied by Neanderthal) that
are on the device.

## Require the Right Namespaces and Set up the Context

Functions for creating the appropriate OpenCL (that means GPU but also other
accellerators) vectors or matrices are in the `uncomplicate.neanderthal.opencl`
namespace.

Import the appropriate namespaces: `core` for computation functions,
`native` for ordinary CPU constructors and `opencl` for the accelerated stuff.
We also need to discover and set up the device, so we need
`uncomplicate.clojurecl.core`.

To be sure that this code is always in the proper working condition,
I'll write it as a bunch of midje test facts and include it in the test suite,
therefore you can ignore these `facts` and `=>`s, they're not part of Neanderthal.

```Clojure

(ns uncomplicate.neanderthal.examples.guides.tutorial-opencl-test
  (:require [midje.sweet :refer [facts => truthy]]
            [uncomplicate.clojurecl.core
             :refer [with-default with-release finish! *command-queue*]]
            [uncomplicate.neanderthal
             [core :refer [asum dot axpy! mv! mm!]]
             [native :refer [sv sge]]
             [opencl :refer [with-default-engine sclv sclge write!]]]))

```

To be able to communicate with the GPU, we need to connect to the device drivers
via the ClojureCL library and create appropriate contexts and queues through which we
can fine tune the computation executions. Neanderthal can also work with the
default setting, which we'll do here because we do not need all the ClojureCL
knobs for the beginning.

First, we will wrap all code we work with into `with-default` and `with-default-engine`.
Our code will then be executed on the first available device on the first available
OpenCL platform. On my machine, that would activate the fastest GPU (the first
of the three Radeons I have) using the AMD's OpenCL drivers. Your setup would probably
be more or less different.

```Clojure

(with-default
  (with-default-engine
    (facts "We'll write our GPU code here, but for now here is only plain
CPU stuff you recognize from the plain Neanderthal tutorial."

           (asum (sv 1 -2 3)) => 6.0)))

```

## Do The Computations

Let's see how to do the same computation on the GPU:

```Clojure

(with-default
  (with-default-engine
    (facts
     "Create a vector on the device, write into it the data from the host vector
and compute the sum of absolute values."

     (with-release [gpu-x (write! (sclv 3) (sv 1 -2 3))]

       (asum gpu-x)) => 6.0)))

```

And that is all you need to begin. That sum has just been computed on your GPU!

I'll make a cup of coffee and spice it up with some chocolate milk
so I can drink it while I am explaining what we have just done here. In the
meantime, you can study that code, and probably figure it out yourself.

_Making the drink..._

Here are the important steps:

1. Create the 3-element empty vector on the GPU device with `sclv`.
That is short for 'single precision floating point CL vector'. We are using
single precision because affordable consumer-grade GPU devices offer amazing
performance with floats, and are much slower (Radeons 290X - 8x, GeForce 980 - 32x)
with doubles. Most of the time you do not even need double precision, but when
you do, you need to shell more $$$ for the professional grade products such as
FirePro and Tesla. Neanderthal supports doubles just the same.

2. Write the data from an ordinary vector `(sv 1 -2 3)` to the GPU. That data
needs to travel from Java to raw memory and from raw memory, over PCIe bus,
to the GPU memory. That is a lot of steps, and Neanderthal does all these with
only one physical copying, but anyway, that data needs to travel and it takes
some precious time, so you should do this as little as possible.

3. Very important: hold the reference to that GPU vector and release it once
you do not need it. Neanderthal can do that bookkeeping for you if you use
ClojureCL's `with-release` macro which works just like Clojure's let form,
but keeps in mind to release the memory on the device when the code reaches
the end or if any exception happens to be thrown. Neanderthal would work without
this, but your GPU memory will fill up after some time and refuse to work further.

4. The happy stuff: **call Neanderthal core functions in the same way you'd do
for the plain CPU Neanderthal vectors and matrices.** _Yes, it is that easy_.

## Measure the Performance

So, what speedups should you expect over native-optimized CBLAS that is Neanderthal's
default? Let's measure it. I'm running this on Intel i7 4790k CPU and AMD
Radeon R9 290x GPU. Your hardware will give different numbers.

```Clojure

  (with-default
    (with-default-engine
      (facts
       "Compare the speed of computing small vectors on CPU and GPU"
       (with-release [host-x (sv 1 -2 3)
                      gpu-x (write! (sclv 3) host-x)]

         (println "CPU:")
         (time (asum host-x)) => 6.0
         (println "GPU:")
         (time (asum gpu-x)) => 6.0))))

```

When measuring very fast code, the `time` function gives wildly imprecise results
- replace the calls to `time` with calls for criterium `quick-bench` and it will
show much faster and precise measurements. Anyway, we can see that CPU is much faster:
28 nanoseconds vs 58 microseconds. This is because calling GPU takes some time
(on the order of magnitude of 20-30 microseconds per kernel enqueue, depending
on the device), and when the device starts computing, most of its many
computing units are idling because we have loaded it with only 3 numbers
to compute, which it does instantly.

Let's try again with more data!

```Clojure

  (with-default
    (with-default-engine
      (facts
       "Let's try with 2^20. That's more than a million."
       (let [cnt (long (Math/pow 2 20))]
         (with-release [host-x (sv (range cnt))
                        gpu-x (write! (sclv cnt) host-x)]

           (println "CPU:")
           (time (asum host-x)) => (float 5.49754798E11)
           (println "GPU:")
           (time (asum gpu-x)) => 5.497552896E11)))))

```

On my machine, it's almost a tie. Criterium reports 99 microseconds on the CPU
vs 107 microseconds on the GPU.

A million is still smallish, though. Let's get serious. Let's give a vector of
2GB (that's 536 million entries) to both:

```Clojure

  (with-default
    (with-default-engine
      (facts
       "Let's try with 2^29. That's 2GB, the maximum that Java buffers can
currently handle. Java 9 would hopefully increase that."
       (let [cnt (long (dec (Math/pow 2 29)))]
         (with-release [host-x (sv (range cnt))
                        gpu-x (write! (sclv cnt) host-x)]

           (println "CPU:")
           ;; note the wrong result in the CPU vector. That's because single precision floats
           are not enough for so many accumulations. In real life, you must use doubles where needed.
           (time (asum host-x)) => (float 1.08086391E17)
           (println "GPU:")
           ;; GPU engine uses doubles for this accumulation, so the result is more precise.
           (time (asum gpu-x)) => 1.44115187270549504E17)))))

```

CPU: 92 milliseconds
GPU: 27 milliseconds

Underwhelming. Is that it? This GPU has 5632 GFLOPS, while the CPU has only 32 or so.
That's 176x more muscle! Should the difference be much bigger? The point is: we
should keep that muscle busy, and we cannot because the computing units
are still idling most of the time waiting data to be transferred from the
device memory to the device registers. Sum is a so simple an operation
that the main constraint is memory throughput, not computing power.

```Clojure


  (with-default
    (with-default-engine
      (facts
       "Let's try with a more parallel linear operation: adding two vectors.
I'll set them to 1GB each because my GPU does not have enough memory to
hold 4GB of data (it has 4GB total memory)."
       (let [cnt (long (Math/pow 2 28))]
         (with-release [host-x (sv (range cnt))
                        host-y (sv (range cnt))
                        gpu-x (write! (sclv cnt) host-x)
                        gpu-y (write! (sclv cnt) host-y)]

           (println "CPU:")
           (time (axpy! 3 host-x host-y)) => host-y
           (println "GPU:")
           (time (do (axpy! 3 gpu-x gpu-y) (finish! *command-queue*))) => truthy)))))

```

CPU: 159 ms
GPU: 41 ms

Still a difference of only 4x. Linear 1D operations are simply so
easy on computation that GPU cannot show its power. They are still useful, though. If
your vector data is already on the GPU, where it participates in some complex
computations that GPU shines at, then it is easier to compute it on the GPU
than to transfer it back and forth to the CPU.

Let's try with some BLAS 2 operation. Their quadratic complexity should matter.
We'll do a matrix - vector multiplication.

```Clojure


  (with-default
    (with-default-engine
      (facts
       "Matrix-vector multiplication. Matrices of 8192x8192 (268 MB) are usually
demanding enough."
       (let [cnt 8192]
         (with-release [host-a (sge cnt cnt (range (* cnt cnt)))
                        host-x (sv (range cnt))
                        host-y (sv (range cnt))
                        gpu-a (write! (sclge cnt cnt) host-a)
                        gpu-x (write! (sclv cnt) host-x)
                        gpu-y (write! (sclv cnt) host-y)]

           (println "CPU:")
           (time (mv! 3 host-a host-x 2 host-y)) => host-y
           (println "GPU:")
           (time (do (mv! 3 gpu-a gpu-x 2 gpu-y) (finish! *command-queue*))) => truthy)))))

```

CPU: 15.4 ms
GPU: 2.77 ms

That's a 5.5x win for the GPU. Nothing too much, but still ok. Let's try matrix
multiplication and see how that goes.

```Clojure


  (with-default
    (with-default-engine
      (facts
       "Matrix-vector multiplication. Matrices of 8192x8192 (268 MB) are usually
demanding enough."
       (let [cnt 8192]
         (with-release [ host-a (sge cnt cnt (range (* cnt cnt)))
                        host-b (sge cnt cnt (range (* cnt cnt)))
                        host-c (sge cnt cnt (range (* cnt cnt)))
                        gpu-a (write! (sclge cnt cnt) host-a)
                        gpu-b (write! (sclge cnt cnt) host-a)
                        gpu-c (write! (sclge cnt cnt) host-a)]

           (println "CPU:")
           (time (mm! 3 host-a host-b 2 host-c)) => host-c
           (println "GPU:")
           (time (do (mm! 3 gpu-a gpu-b 2 gpu-c) (finish! *command-queue*))) => truthy))))))

```

CPU: 17678 ms
GPU: 721 ms

That's 25 times faster than the CPU! But, still, shouldn't it be even faster?
You've probably seen those benchmarks with 1000x speed improvements!

Let's consider matrix multiplication. It is a complex operation - O(n^3), but at
its core a very simple computation. For each few float multiplications and additions
there is also a couple of memory readings and writings. Therefore, GPU wins
hugely, but it still has unused computation power.

## Thinking About the Results

**Where do those 1000x numbers come from then? That depends on what you compare to.**

**This is a very important issue. Remember, here we've compared Neanderthal's GPU
speed to Neanderthal's highly optimized native ATLAS BLAS engine, which is
a speed demon in its own right! And we got a 25x speedup.**

**How does Neanderthal compare to pure Java? Check out the [Neanderthal Benchmarks page](benchmarks.html).
For 8192x8192 matrices, an optimized and decently fast pure Java library Vectorz
(which is the core.matrix flagship implementation)
working with primitives and optimizing cache use, needs 6.14 minutes to compute.
That's 368400 milliseconds. Neanderthal GPU is 510x faster than that! And, there
are several GPUs on the market that are considerably faster than my Radeon 290X.**

**Of course, if you try to write your own nested loops to compute these matrices,
even pure Java libraries will run circles around your operations, and Neanderthal
will be a thousand or even several thousands times faster, even when you write
tight Java loops with primitives.**

**Matrix algebra is only a start. The real benefit is when you use Neanderthal as
a gate and a foundation to write your own ClojureCL numerical computation kernels
for your own number crunching algorithms. If they are computationally intensive
enough and parallel, THEN you can hope for real thousandfold improvements.**

## Can You Run This On Your Own Machine?

At the time of writing of this text, Neanderthal builds its accelerator support
on ClojureCL, which is a Clojure library for programming with OpenCL, which is in
turn an open standard for GPU and accelerated computing. Thanks to Neanderthal's
pluggable architecture, BLAS engines optimized for any OpenCL-compatible
architecture can be transparently added, and the code of this tutorial does not
need to change at all.

However, be aware that the first implementation available today is OpenCL 2.0 only
and optimized for AMD's GCN architecture (R9 Radeons, newer FirePro). I am running this
tutorial on AMD Radeon R9 290X. It should work on Intel hardware, but I doubt
that the performance would be even remotely close. Nvidia currently supports
only OpenCL 1.2. Providing optimized Neanderthal engines for Nvidia and Intel is
planned, but will not be immediate. If you would like to do superfast numerical
computations in Clojure soon, and do not wish to spend $350 for an AMD GPU such
as the one that I have, you might even contribute a BLAS engine implementation
for your chosen architecture! :)

Happy hacking!

```Clojure

(facts "Are you ready to write number crunching Clojure code now?"
       :answer => truthy)
