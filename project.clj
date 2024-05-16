;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/neanderthal "0.49.1"
  :description "Neanderthal is a Clojure library for fast matrix and linear algebra computations."
  :url "https://github.com/uncomplicate/neanderthal"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/neanderthal"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.3"]
                 [uncomplicate/commons "0.15.1"]
                 [uncomplicate/fluokitten "0.10.0"]
                 [org.uncomplicate/clojure-cpp "0.3.0"]
                 [org.bytedeco/mkl-platform "2024.0-1.5.10"]
                 [uncomplicate/clojurecl "0.16.0"]
                 [org.jocl/jocl-blast "1.5.2"]
                 [uncomplicate/clojurecuda "0.19.0"]
                 [org.apache.commons/commons-math3 "3.6.1"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [lein-codox "0.10.8"]
                             [com.github.clj-kondo/lein-clj-kondo "0.2.5"]]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.10.10"]
                                  [codox-theme-rdash "0.1.2"]
                                  [org.bytedeco/mkl "2024.0-1.5.10" :classifier linux-x86_64-redist]
                                  [org.bytedeco/cuda "12.3-8.9-1.5.10" :classifier linux-x86_64-redist]]
                   :codox {:metadata {:doc/format :markdown}
                           :source-uri "http://github.com/uncomplicate/neanderthal/blob/master/{filepath}#L{line}"
                           :themes [:rdash]
                           :namespaces [uncomplicate.neanderthal.auxil
                                        uncomplicate.neanderthal.block
                                        uncomplicate.neanderthal.core
                                        uncomplicate.neanderthal.cuda
                                        uncomplicate.neanderthal.integer
                                        uncomplicate.neanderthal.linalg
                                        uncomplicate.neanderthal.math
                                        uncomplicate.neanderthal.native
                                        uncomplicate.neanderthal.opencl
                                        uncomplicate.neanderthal.random
                                        uncomplicate.neanderthal.real
                                        uncomplicate.neanderthal.vect-math
                                        uncomplicate.neanderthal.sparse]
                           :output-path "docs/codox"}

                   ;;also replaces lein's default JVM argument TieredStopAtLevel=1
                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"]}}

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :classifiers {:tests {:source-paths ^:replace ["test"]}}
  :source-paths ["src/clojure" "src/device"]
  :java-source-paths ["src/java"]
  :test-paths ["test"])
