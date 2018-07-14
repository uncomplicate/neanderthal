 "
---
title: 'Tutorial - Matrix Computations on the GPU in Clojure (in TFLOPS!)'
Author: Dragan Djuric
layout: article
---

So, you've [installed Neanderthal](getting_started.html) and [learned
how to work](tutorial_native.html) with vectors and matrices on the CPU at 100X
 the speed of pure Java libs. It is fast, running at hundreds of GFLOPS,
but you've read that nowadays the computations on GPUs are what all the hipster
geeks do, so you wonder whether there is something to it. If only you could
connect to [CUBLAS](https://developer.nvidia.com/cuBLAS), your algorithms would
speed up thousandfold with almost no sweat, as in commercials...

I have some news for you:

* The bad: it is not that simple. CPU algorithms probably suck on massively
parallel architectures, and you'd need to learn quite a few new tricks to collect
the benefits you see on NVIDIA and AMD websites.

* The good: Neanderthal implements BLAS algorithms and abstracts most of that
complexity for vector and matrix computations away, behind a friendly Clojure API.

**TL/DR: Multiplication of large matrices is more than 1000x faster with Neanderthal
than with optimized pure Java libraries, 10x faster than Neanderthal native engine
and many thousands of times faster than the nested looping code with primitives
that you'd write yourself.**

## What You Need to Know About GPU Programming Before We Begin

The most important thing when working with parallel accellerators (GPUs and others)
is to be aware that a large part of your code will run on a physically separate
device, which also has its own working memory. It can not access data from the
main memory, such as your objects, arrays, primitives and such. It also can not
run your arbitrary program code (Java, C, R). Even if/when it could (Aparapi)
it sucks at it big time, because it is made of thousands of very dumb processors that
are excellent at one thing only - raw computations - and poor at everything else,
including logic.

So, a typical approach is that your program consists of two parts: host and device.
Host is a typical Clojure (Java) program that do the usual things Clojure programs
do: talks to the web, formats and extracts data, does the database dance,
recursively compute factorials, and perform other logic-heavy tasks.
Then, when it has collected the data and stuffed it in a
big raw chunk of bazillion primitive floating point numbers it sends it to the
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
it is absolutely necessary. Even avoid the communication with the device unless
it is absolutely necessary.

The typical workflow would be: prepare the input data in your Clojure program and
send it to the device. Then, call many Neanderthal and ClojureCL functions that
work with that data without transferring it back. Only transfer the final result.

You'll still write your algorithms in Clojure as any normal Clojure code,
you just have to be aware that the data is actually on the device, and that the
Neanderthal functions you call run the kernels (supplied by Neanderthal) that
are on the device.

## Require the Right Namespaces and Set up the Context

Functions for creating the appropriate OpenCL (that means GPU but also other
accelerators) vectors or matrices are in the `uncomplicate.neanderthal.opencl`
namespace.

Import the appropriate namespaces: `core` for computation functions,
`native` for ordinary CPU constructors and `opencl` for the accelerated stuff.
We also need to discover and set up the device, so we need
`uncomplicate.clojurecl.core`.

And, to be sure that this code is always in the proper working condition,
I'll write it as a bunch of midje test facts and include it in the test suite,
therefore do not mind these `facts` and `=>`s, they're not part of Neanderthal.

$code"


(ns uncomplicate.neanderthal.examples.guides.tutorial-opencl-test
  (:require [midje.sweet :refer [facts => truthy roughly]]
            #_[criterium.core :refer [quick-bench with-progress-reporting]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.clojurecl
             [core :refer [finish! with-default-1]]]
            [uncomplicate.neanderthal
             [core :refer [asum dot axpy! mv! mm! transfer! copy]]
             [native :refer [fv fge]]
             [opencl :refer [with-default-engine clv clge]]]))

"$text

To be able to communicate with the GPU, we need to connect to the device drivers
via the ClojureCL library and create appropriate contexts and queues through which we
can fine tune the computation executions. Neanderthal can also work with the
default setting, which we'll do here because we do not need all the ClojureCL
knobs for the beginning.

So, we will wrap all code we work with into `with-default` and `with-engine`.
Our code will then be executed on the first available device on the first available
OpenCL platform. On my machine, that would activate the most capable GPU
using the AMD's OpenCL drivers. Your setup may be different.

$code"

(with-default-1
  (with-default-engine
    (facts "We'll write our GPU code here, but for now here is only the plain
CPU stuff you recognize from the plain Neanderthal tutorial."

           (asum (fv 1 -2 3)) => 6.0)))

"$text

## Do The Computations

Let's see how to do the same computation on the GPU:

$code"

(with-default-1
  (with-default-engine
    (with-release [gpu-x (transfer! (fv 1 -2 3) (clv 3))]
      (facts
       "Create a vector on the device, write into it the data from the host vector
and compute the sum of absolute values."
       (asum gpu-x) => 6.0))))

"$text

And that is all you need to begin. That sum has just been computed on your GPU!

I'll make a cup of coffee and spice it with some chocolate milk
so I can drink it while I am explaining what we have just done here. In the
meantime, you can study that code, and probably figure it out yourself.

`<Preparing the drink... >`

Here are the important steps:

1. Create the 3-element empty vector on the GPU device with `clv`.
That is short for 'single precision floating point CL vector'. We are using
single precision because affordable consumer-grade GPU devices offer amazing
performance with floats, and are much slower (Radeons 290X - 8x, GeForce 980 - 32x)
with doubles. Most of the time you do not even need double precision, but when
you do, you need to shell more $$$ for the professional grade products such as
FirePro and Tesla. Neanderthal supports doubles just the same.

2. Write the data from an ordinary vector `(fv 1 -2 3)` to the GPU. That data
needs to travel from Java to raw memory and from raw memory, over PCIe bus,
to the GPU memory. That is a lot of steps, and Neanderthal does all these with
only one physical copying, but anyway, that data needs to travel and it takes
some precious time, so should you do this as rarely as possible.

3. Very important: hold the reference to that GPU vector and release it once
you do not need it. Neanderthal can do that bookkeeping for you if you use
ClojureCL's `with-release` macro which works just like Clojure's let form,
but keeps in mind to release the memory on the device when the code reaches
the end or if any exception happens to be thrown. Neanderthal would work without
this, but your GPU memory will fill up after some time, and refuse to work further.

4. The happy stuff: **call Neanderthal core functions in the same way you'd do
for the plain CPU Neanderthal vectors and matrices.** _Yes, it is that easy_.

## Measure the Performance

So, what speedups you should expect over native optimized CBLAS that is Neanderthal's
default? Let's measure it. I'm running this on Intel i7 4790k CPU and AMD
Radeon R9 290x GPU. Your hardware will give different numbers.

$code"

(with-default-1
  (with-default-engine
    (with-release [host-x (fv 1 -2 3)
                   gpu-x (clv 1 -2 3)]
      (facts
       "Compare the speed of computing small vectors on CPU and GPU"
       (asum host-x) => 6.0
       #_(println "CPU:")
       #_(with-progress-reporting (quick-bench (asum host-x)))
       (asum gpu-x) => 6.0
       #_(println "GPU:")
       #_(with-progress-reporting (quick-bench (do (asum gpu-x) (finish!))))))))

"$text

When measuring very fast code, the `time` function gives wildly imprecise results
- thus we replace the calls to `time` with calls for criterium `quick-bench` and it
shows much faster and precise measurements. Anyway, we can see that CPU is much faster:
37 nanoseconds vs many microseconds. This is because calling GPU takes some time
(on the order of magnitude of 20-30 microseconds per kernel enqueue, depending
on the device), and when the device starts computing, most of its many
computing units are idling because we have loaded it with only 3 numbers
to compute, which it does instantly. When measuring GPU calls, we add
the call to `finish!` to make sure we wait for the computation to actually
be completed. If we didn't do this, we'd measure only the time it takes
to tell the GPU to do the computation.

Let's try again with more data!

$code"

(with-default-1
  (with-default-engine
    (let [cnt (long (Math/pow 2 20))]
      (with-release [host-x (fv (range cnt))
                     gpu-x (transfer! host-x (clv cnt))]
        (facts
         "Let's try with 2^20. That's more than a million."

         (asum host-x) => (float 5.49755322E11)
         #_(println "CPU:")
         #_(with-progress-reporting (quick-bench (asum host-x)))

         (asum gpu-x) => 5.497552896E11
         #_(println "GPU:")
         #_(with-progress-reporting (quick-bench (do (asum gpu-x) (finish!)))))))))

"$text

On my machine, it's closer, but CPU is still faster. Criterium reports 29 microseconds on the CPU
 vs 102 microseconds on the GPU with clblast engine.

A million is still smallish, though. Let's get serious. Let's give a vector of
1GB (that's 536 million entries) to both:

$code"

#_(with-default-1
  (with-default-engine
    ;; I had to change it to 2^28 because a recent update for my GPU driver caused
    ;; it to complain about insufficient memory, but this is probably a temporary issue.

    (let [cnt (long (Math/pow 2 28))]
      (with-release [host-x (fv (range cnt))
                     gpu-x (transfer! host-x (clv cnt))]
        (facts
         "Let's try with 2^28. That's 1GB, half the maximum that Java buffers can
currently handle. Java 9 would hopefully increase that."

         ;; note the less precise result in the CPU vector. That's because single
         ;; precision floats are not precise enough for so many accumulations.
         ;; In real life, sometimes you must use doubles in such cases.
         (asum host-x) => (roughly 3.6077906E16)
         #_(println "CPU:")
         #_(with-progress-reporting (quick-bench (asum host-x)))

         ;; GPU engine uses doubles for this accumulation, so the result is more precise.
         (asum gpu-x) => (roughly 3.60287949E16)
         #_(println "GPU:")
         #_(with-progress-reporting (quick-bench (do (asum gpu-x) (finish!)))))))))

"$text

CPU: 37 milliseconds
GPU: 8 milliseconds

Underwhelming. Is that it? This GPU has 5632 GFLOPS, while the CPU has only 32 per core or so.
That's 176x more muscle! Should the difference be much bigger? The point is: we
should keep that muscle busy, and we can not, because the computing units
are still idling most of the time waiting data to be transferred from the
device memory to the device registers. Sum is a rather simple operation,
that the main constraint is memory throughput, not computing power.

$code"

#_(with-default-1
  (with-default-engine
    (let [cnt (long (Math/pow 2 28))]
      (with-release [host-x (fv (range cnt))
                     host-y (copy host-x)
                     gpu-x (transfer! host-x (clv cnt))
                     gpu-y (copy gpu-x)]
        (facts
         "Let's try with a more parallel linear operation: adding two vectors.
I'll set them to 1GB each because my GPU does not have enough memory to
hold 4GB of data (it has 4GB total memory)."

         (axpy! 3 host-x host-y) => host-y
         #_(println "CPU:")
         #_(with-progress-reporting (quick-bench (axpy! 3 host-x host-y)))

         (axpy! 3 gpu-x gpu-y) => gpu-y
         #_(println "GPU:")
         #_(with-progress-reporting (quick-bench (do (axpy! 3 gpu-x gpu-y) (finish!)))))))))

"$text

CPU: 118 ms
GPU: 11 ms

Not bad, but still only 10x faster. Linear 1D operations are simply so
easy on computation that GPU can not show it's power. They are still useful, though. If
your vector data is already on the GPU, where it participates in some complex
computations that GPU shines at, **then it is easier and faster to compute it on the GPU
than to transfer it back and forth to the CPU**.

Let's try with some BLAS 2 operation. Their quadratic complexity should matter.
We'll do a matrix - vector multiplication.

$code"

(with-default-1
    (with-default-engine
      (let [cnt 8192]
        (with-release [host-a (fge cnt cnt (range (* cnt cnt)))
                       host-x (fv (range cnt))
                       host-y (copy host-x)
                       gpu-a (transfer! host-a (clge cnt cnt))
                       gpu-x (transfer! host-x (clv cnt))
                       gpu-y (copy gpu-x)]
          (facts
           "Matrix-vector multiplication. Matrices of 8192x8192 (268 MB) are usually
demanding enough."

           (mv! 3 host-a host-x 2 host-y) => host-y
           #_(println "CPU:")
           #_(with-progress-reporting (quick-bench (mv! 3 host-a host-x 2 host-y)))

           (mv! 3 gpu-a gpu-x 2 gpu-y) => gpu-y
           #_(println "GPU:")
           #_(with-progress-reporting (quick-bench (do (mv! 3 gpu-a gpu-x 2 gpu-y) (finish!)))))))))

"$text

CPU: 9.6 ms
GPU: 1.01 ms

That's a 9x win for the GPU. Nothing too much, but still ok. Let's try matrix
multiplication and see how that goes.

$code"

#_(with-default-1
    (with-default-engine
      (let [cnt 8192]
        (with-release [host-a (fge cnt cnt (range (* cnt cnt)))
                       host-b (copy host-a)
                       host-c (copy host-a)
                       gpu-a (transfer! host-a (clge cnt cnt))
                       gpu-b (copy gpu-a)
                       gpu-c (copy gpu-a)]
          (facts
           "Matrix-matrix multiplication. Matrices of 8192x8192 (268 MB) are usually
demanding enough."

           (println "CPU:")
           (time (mm! 3 host-a host-b 2 host-c)) => host-c
           (mm! 3 gpu-a gpu-b 2 gpu-c) => gpu-c
           (finish!)
           (println "GPU:")
           (time (do (mm! 3 gpu-a gpu-b 2 gpu-c) (finish!))))))))

"$text

CPU: 3157 ms
GPU: 293 ms

Note: Nvidia GTX 1080: 220 ms.

That's almost 11x faster than the CPU working in 4 threads! But, still, shouldn't it be even faster?
You've probably seen those benchmarks with 1000x speed improvements!

Let's consider matrix multiplication. It is a complex operation - O(n^3), but at
its core a very simple computation. For each few float multiplications and additions
there is also a couple of memory readings and writings. Therefore, GPU wins
hugely, but it still has unused computation power.

## Thinking About the Results

**Where do those 1000x numbers come from then? That depends on what you compare to.**

This is a very important issue. Remember, here **we've compared Neanderthal's GPU
speed to Neanderthal's highly optimized native MNKL BLAS engine, which is
a speed demon in its own right! And we got a 10x speedup.** If you take the fastest
Intel Xeon with dozens of cores, costing thousands of dollars, you might even approach
the speed of a consumer GPU costing $300.

**How it stands to pure Java?** Check out the [Neanderthal Benchmarks page](benchmarks.html).
For 8192x8192 matrices, an optimized and decently fast pure Java library Vectorz
(which is the core.matrix flagship implementation)
working with primitives and optimizing cache use, needs 6.14 minutes to compute.
That's 368400 milliseconds. Neanderthal GPU is 1250x faster than that (on the rather old AMD R9 290X,
and **1675x** on the newer Nvidia GTX 1080)! And, there
are several GPUs on the market that are considerably faster than my Radeon 290X.

Of course, if you try to write your own nested loops to compute these matrices,
even pure Java libraries will run circles around your operations, and **Neanderthal
will be several thousands times faster**, even when you write tight Java loops with primitives.

Matrix algebra is only a start. **The real benefit is when you use Neanderthal as
a gateway and a foundation to write your own ClojureCL numerical computation kernels
for your own number crunching algorithms. If they are computationally intensive
enough and parallel, THEN you can hope for real thousandfold improvements.**

## Can You Run This On Your Own Machine?

**Absolutely! It works on any sufficiently modern OpenCL platform, which includes
AMD, Nvidia, and Intel!** Please note that the speed will depend on the actual hardware,
so if you intend to use a laptop GPU, you might not be impressed with the speed.

Happy hacking!

$code"

(facts "Are you ready to write number crunching Clojure code now?"
       :answer => truthy)
