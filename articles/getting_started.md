---
title: "Getting Started"
Author: Dragan Djuric
layout: article
---
# Get Started
This is a brief introductory guide to Neanderthal that aims to give you the necessary information to get up and running, as well as a brief overview of some available resources for learning about matrix computations and apply them in Clojure with Neanderthal.

## How to get started
* Walk through this guide, set up your development environment, and try the examples.
* Familiarize yourself with Neanderthal's [more advanced guides](/articles/guides.html#neanderthal_documentation_and_tutorials) and [API documentation](/codox).

## Overview and features

Neanderthal is a Clojure library for fast matrix and linear algebra computations based on the highly optimized ATLAS BLAS and LAPACK native library. It provides the following features:

### Already implemented

* Data structures: double vector, double general dense matrix (GE);
* BLAS Level 1, 2, and 3 routines;
* Various Clojure vector and matrix functions (transpositions, submatrices etc.);
* Fast map, reduce and fold implementations for the provided structures.

### On the TODO list

* LAPACK routines;
* Banded, symmetric, triangular matrices;
* Support for complex numbers;
* Support for single-precision floats.;

## Installation

Neanderthal is a Clojure library packaged in two `jar` files, distributed through [Clojars](http://clojars.org). One is a pure Clojure library that you will use directly, and the other contains native JNI bindings for a specific operating system. They follow [maven](http://www.maven.org) (and [leiningen](http://www.leiningen.org)) naming conventions:

* Pure Clojure lib: `groupId` is `uncomplicate` and `artifactId` is `neanderthal`.
* Native JNI lib: `groupId` is `uncomplicate` and `artifactId` is `neanderthal-atlas` with a classifier for your operating system and architecture, e.g. `amd64-Linux-gpp-jni`.

Neanderthal also **uses the native ATLAS library and expects that you make it available on your system, typically as a shared so, dll, or dylib!** ATLAS is highly optimized for various architectures - if you want top performance **you have to build ATLAS from the source**. Do not worry, ATLAS comes with automatic autotools build script, and a detailed configuration and installation guide. If you do not follow this procedure, and use a pre-packaged ATLAS provided by your system (if it exists), you will probably get very degraded performance compared to a properly installed ATLAS.

### With Leiningen

The most straightforward way to include Neanderthal in your project is with leiningen. Add the following dependency to your `project.clj`:

```clojure
[uncomplicate/neanderthal "0.3.0"]
```

### With Maven

Add Clojars repository definition to your pom.xml:

```xml
<repository>
  <id>clojars.org</id>
  <url>http://clojars.org/repo</url>
</repository>
```

And then the dependency:

```xml
<dependency>
  <groupId>uncomplicate</groupId>
  <artifactId>neanderthal</artifactId>
  <version>0.3.0</version>
</dependency>
```

### Requirements

Neanderthal requires at least Clojure `1.5`, as it uses the reducers library.
Reducers use java fork/join, which requires Java 7+ jdk, or Java 6 with `jsr166y.jar` included in project dependencies (see [Clojure's POM] (https://github.com/clojure/clojure/blob/master/pom.xml) for the dependency info). Neanderthal has no other dependencies or requirements, excepting some non-mandatory test helpers that need the [midje](https://github.com/marick/Midje) library to pass the testing suite.

## Usage

[See the this tutorial's source as a midje test.](https://github.com/uncomplicate/neanderthal/blob/master/test/uncomplicate/neanderthal/articles/getting_started_test.clj)

First `use` or `require` `uncomplicate.neanderthal.core` and `uncomplicate.neanderthal.jvm` in your namespace, and you'll be able to call appropriate functions from the Neanderthal library.

```clojure
(ns example
  (:use [uncomplicate.neanderthal core jvm]))
```

What functions does this make available? Not many (which is a good thing), because a key point of CT programming is to utilize a small set of highly generalized and well defined operators and combinators that can be mixed and combined simply to achieve desired outcomes. The following demonstrates simple examples, mostly using Neanderthal extensions to Clojure built-in constructs.

### Functors and fmap

The basic categorical concepts concern contexts for data. One familiar data context is a sequence, for example one that represents several numbers. Suppuse we have a vanilla function that operates on numbers and we want to apply it to data without regard to the wrapper (context) that contains it. Of course, Clojure already has a function that can reach inside the sequence and apply a function to each element -- the ubiquitous `map` function:

```clojure
(map inc [1 2 3])
;=> (2 3 4)

```

The only, usually minor, problem with `map` is that it outputs a lazy sequence, so our context is damaged a bit -- we start with a vector and end with a sequence.

Let's try an alternative - the `fmap` function:

```clojure
(fmap inc [1 2 3])
;=> [2 3 4])
```

It's similar to `map`, but the starting context is preserved. `fmap` reaches inside any context that implements the `Functor` protocol (in this case, a Clojure vector), applies a plain function (here, `inc`) to the data inside and produces a result inside the same type of context.
Neanderthal extends all standard Clojure collections with the `Functor` protocol, and provides specific implementations of fmap for each, as we see below. Note that, depending on how many arguments the function can accept, we may provide many contexts to `fmap`.

```clojure
(fmap + [1 2 3] [1 2 3 4])
;=> [2 4 6]

(fmap + (list 1 2 3) [1 2] #{1 2 3 4})
;=> (3 6)

(fmap + {:a 1 :b 2} {:a 3 :c 4} {:d 5})
;=> {:a 4 :b 2 :c 4 :d 5}
```

Of course, Clojure collections are not the only implementations of the `Functor` protocol. Neanderthal extends most of the Clojure types with the appropriate implementations of `Functor` protocol. For example:

```clojure
(fmap * (atom 2) (ref 3) (atom 4))
;=> (atom 24)

((fmap inc *) 2 3)
;=> 7
```

Of course, you can also build your own implementations, which is covered in [detailed guides](/articles/guides.html).

### Applicative functors: pure and fapply

Starting with the same idea of data inside a context, we can extend it to the function part: what if we want to apply plain functions that are wrapped in context to data in similar contexts? For example, a vector of functions applied to vector(s) of data. For this purpose Neanderthal provides `fapply`:

```clojure
(fapply [inc dec (partial * 10)] [1 2 3])
;=> [2 3 4 0 1 2 10 20 30]
```

`fapply` reaches inside any `Applicative` context (in this case, vector), applies function(s) wrapped in the same type of context (vector) and produces a similarly wrapped result. As an `Applicative`, vector produces a context wrapping results of applying all wrapped functions to all wrapped data.
More simple examples:

```clojure
(fapply [+ -] [1 2] [3 4])
;=> [4 6 -2 -2]

(fapply {:a + :b *} {:a 1 :b 2} {:a 3 :b 3 :c 4} {:d 5})
;=> {:a 4, :b 6, :c 4, :d 5}
```

`Applicative`s also support a function that wraps any data into minimal typed context -- `pure`:

```clojure
(pure [] 3)
;=> [3]

(pure (atom nil) 5)
;=> (atom 5)
```

Are these function definitions and implementations arbitrary? NO! All these functions have to satisfy mathematical laws, ensuring they mesh seemlessly with the rest of the framework. Discussion of these laws is well beuond our scope, but you may rest assured that they are rigoroufly followed in the Neanderthal implementation. When you move beyond using the provided contexts and functions to writing youw own CT inplementations, you'll have to become familiar with these laws, which are covered in the advanced guides.

### Monads and bind

Monads are certainly the most discussed programming concept to come from category theory. Like functors and applicatives, monads deal with data in contexts. However, in addition to applying funtions contextually, monads can also transform context by unwrapping data, applying functions to it and rewrapping it, possibly in a completely different context. Sound confusing? Until you gain some practical experience, it is -- that is why monad tutorials are written every day. Don't be daunted, however. If you take a step-by-step approach and don't try to swallow everything in one sitting, it's really not hard at all. This tutorial only scratches the surface; please check out the further reading for deeper comprehension.
The core monad function is `bind`, and in the case of vector, it is trivially used as follows.

```clojure
(bind [1 2 3] #(return (inc %) (dec %)))
;=> [2 0 3 1 4 2]
```

If the function produces minimal context, it does even need to know which context it is. The return function is going to create the right context for the value, in this case atom.

```clojure
(bind (atom 1) (comp return inc))
;=> (atom 2)
```

Neanderthal implements the `Monad` protocol for many Clojure core types. Please check out the tutorials and docs and be patient until it clicks for you.

### Monoids

`Monoid` is a protocol that offers a default operation `op` on some type, and an identity element, `id` for that operation. `op` has to be closed, meaning (op x y) must have the same type as x and y, and it has to be associative. For example, the default operation for numbers in Neanderthal is +, with the identity element 0, while for lists it is concat, with the default element empty list.

```clojure
(id 4)
;=> 0

(op 1 2)
;=> 3

(id [4 5 6])
;=> []

(op [1 2] [3])
;=> [1 2 3]
```

### Foldables and fold

Having seen the some manipulation of contexts and data, we'd like some methods to get it back out, without writing custom, context-specific code. If we implement the `Foldable` protocol, which Neanderthal does for many Clojure types, we can use `fold` function to get a summary of the contextual data:

```clojure
(fold (atom 3))
;=> 3
```
With more than one value though, fold aggregates. If the data are subject to a monoid, the accumulating `op` will produce the folded value:

```clojure
(fold [])
;=> nil

(fold [1 2 3])
;=> 6

(fold [[1 2 3] [3 4 5 4 3] [3 2 1]])
;=> [1 2 3 3 4 5 4 3 3 2 1]

(fold (fold [[1 2 3] [3 4 5 4 3] [3 2 1]]))
;=> 31
```

## Where to go next

Hopefully this guide got you started and now you'd like to learn more. I expect to build a comprehensive base of articles and references for exploring this daunting topic, so please check the [All Guides](/articles/guides.html) page from time to time. More importantly, I will post articles with Clojure code for related articles, tutorials and videos, which use another reference language (Haskell) to discuss category theory. Of course, you should also check Neanderthal API for specific details, and feel free to take a gander at the source while you are there.
