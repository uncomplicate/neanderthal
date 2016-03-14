---
title: "Why You Might Need Neanderthal Even Without Matrices"
Author: Dragan Djuric
layout: article
---

Neanderthal is a Clojure library *[written in Clojure](https://github.com/uncomplicate/neanderthal/tree/master/src/clojure/uncomplicate/neanderthal)* that leverages the best optimized native BLAS implementations (ATLAS) to offer fast vectorized operations in idiomatic Clojure. It also supports [GPU computing](articles/tutorial_opencl.html) for even faster number crunching.

On top of that, let's see how it can be useful for fast number-crunching in pure Clojure, even without native bindings, and even when you can't express your algorithms in built-in vector and matrix operations. The source code for this article can be found [here](https://github.com/uncomplicate/neanderthal/blob/master/examples/benchmarks/src/benchmarks/map_reduce.clj).

Let's say we have a sequence of 100,000 numbers, and would like to compute its elements in some way. In Clojure, we mainly do this with `map` and `reduce` higher-order functions. We are not interested in actual results here, but in speed, so all evaluations are wrapped in Criterium-s `quick-bench`.

``` clojure
(def n 100000)

(def cvx (vec (range n)))
(def cvy (vec (range n)))

(with-progress-reporting (quick-bench
  (reduce + (map * cvx cvy))))

;; Execution time mean : 11.493054 ms

(with-progress-reporting (quick-bench
  (reduce + cvx)))

;; Execution time mean : 970.238233 µs
```

This code computes dot product, and a sum of these sequences. Dot product computes in 11.493054 ms, and sum in 970.238233 µs. The second is even quite optimized since it uses Clojure reducers. That might be fast enough for many typical Clojure applications, since the other, non-numeric, parts of the application are much slower than that. However, numerical performance is at the center of today's hot areas such as  machine learning, data analysis, and deep learning. Clojure's persistent data structures shine in many areas, but numerical computing is not one of them. They use boxed numbers that are much slower than primitive numbers. Let's compare that, and try primitive Java arrays:

``` clojure

(def cax (double-array (range n)))
(def cay (double-array (range n)))

(with-progress-reporting
  (quick-bench
   (areduce ^doubles cax i ret 0.0
            (+ ret (* (aget ^doubles cax i)
                      (aget ^doubles cay i))))))
;; Execution time mean : 141.939897 µs

```

The dot product algorithm was roughly 80 times faster when the data is in primitive Java arrays, and primitive arithmetic operations are used. That's much better! The trouble is, arrays are clunky, and require special macros (`areduce` in this case). In simple examples like dot product it can be tolerable, but for anything more complex, it gets unwieldy.

One partial remedy is to use hip-hip array library. Its macros are a bit more pleasant than Clojure array built-ins:

``` clojure
(with-progress-reporting
  (quick-bench
   (hiphip/asum [x cax y cay] (* x y)))))
;; Execution time mean : 91.268532 µs

(with-progress-reporting
  (quick-bench
   (areduce ^doubles cax i ret 0.0
            (+ ret (aget ^doubles cax i)))))
;; Execution time mean : 92.133214 µs

```

The performance is on the level of what's possible in pure Java, so we'll use these results as a reference. The Code is fine, but the problem is that we have to use macros. That means that we cannot use all the Clojure higher-order function goodies.

So, can we have the best of both worlds? The speed of primitive arrays (or even faster) with all the niceties of Clojure's higher order functions such as `map` and `reduce`? Fortunately, we can!

Neanderthal implements Fluokitten's interfaces that abstract functors, monoids, etc. You'd probably avoid another Haskell-ish brainwashing tutorial, and I'm with you, so I'll present it thorough Clojure's `map` and `reduce`. It's Fluokitten's `fmap`, `fold`, and `foldmap`,  which are almost like `map` and `reduce` but for any Object, and more powerful. So, Neanderthal vectors (and matrices) can be mapped over and reduced in various ways.

First, we create two Neanderthal vectors, and use the built-in `dot`, just to see how fast we can go on one CPU core with doubles:

``` clojure
(def nx (dv (range n)))
(def ny (dv (range n)))

(with-progress-reporting (quick-bench (dot nx ny)))
;; Execution time mean : 32.828791 µs

```
Roughly 3 times faster than in Java. We'll leave it at that, since this is not the topic of this tutorial. We'll pretend that `dot` is some custom function not supported in Neanderthal. First, try this:

``` clojure
(fold (fmap * nx ny))
;; => ClassCastException clojure.core$_STAR_ cannot be cast to clojure.lang.IFn$DDD  uncomplicate.neanderthal.impl.buffer-block/vector-fmap* (buffer_block.clj:349)
```

Neanderthal complains, because it explicitly checks that we do not shoot our algorithm in the hip by providing a function that (potentially) does (un)boxing. It explicitly asks for a type-hinted function. While we are at that, let's also mention that `fmap` would require an intermediate vector for products, which would consume additional memory and slow the algorithm down a bit. That's why we are going to:

* Use primitive type-hinted functions
* Use Fluokitten's `foldmap` which is a function that first maps and then folds (reduces), without requiring intermediate copy of a vector.

``` clojure
(def nx (dv (range n)))
(def ny (dv (range n)))

(defn p+ ^double [^double x ^double y]
  (+ x y))

(defn p* ^double [^double x ^double y]
  (* x y))

(defn sqr ^double [^double x]
  (* x x))

(with-progress-reporting (quick-bench (foldmap p+ 0.0 p* nx ny)))
;; Execution time mean : 186.881027 µs

(with-progress-reporting (quick-bench (fold nx)))
;; Execution time mean : 95.081694 µs

```

That's not a bad result for a custom, pure Clojure algorithm, that offers all niceties of higher-order functions. `fold`, `fmap`, `fmap!` and `foldmap` also have varargs versions, and some other niceties. Please read more about that at [Fluokitten's guides page](http://fluokitten.uncomplicate.org/articles/guides.html).

Fluokitten even makes those functions available for Clojure's persistent data structures. The main difference compared to Clojue's versions is that it does not convert everything to a sequence, varargs versions, and that hey are optimized for each data structure type.

For example, the same old boxed Clojure persistent vectors will be 5 times faster than with ordinary `map` and `reduce` in our example.

``` clojure
(with-progress-reporting (quick-bench (foldmap p+ 0.0 p* cvx cvy)))
;; Execution time mean : 2.316850 ms

```

As for the Neanderthal, the right place to proceed is [Neanderthal Guides Page](/articles/guides.html).
