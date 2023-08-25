;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/neanderthal "0.48.0-SNAPSHOT"
  :description "Neanderthal is a Clojure library for fast matrix and linear algebra computations."
  :url "https://github.com/uncomplicate/neanderthal"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/neanderthal"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [uncomplicate/commons "0.14.0-SNAPSHOT"]
                 [uncomplicate/fluokitten "0.9.1"]
                 [org.uncomplicate/clojure-cpp "0.2.0-SNAPSHOT"]
                 [org.bytedeco/mkl-platform "2023.1-1.5.10-SNAPSHOT"]
                 [uncomplicate/clojurecl "0.15.2-SNAPSHOT"]
                 [org.jocl/jocl-blast "1.5.2"]
                 [uncomplicate/clojurecuda "0.18.0-SNAPSHOT"]
                 [org.apache.commons/commons-math3 "3.6.1"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [lein-codox "0.10.7"]]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.10.9"]
                                  [codox-theme-rdash "0.1.2"]
                                  [org.bytedeco/mkl-platform-redist "2023.1-1.5.10-SNAPSHOT"]
                                  [org.bytedeco/cuda-platform-redist "12.1-8.9-1.5.10-SNAPSHOT"]]
                   :codox {:metadata {:doc/format :markdown}
                           :source-uri "http://github.com/uncomplicate/neanderthal/blob/master/{filepath}#L{line}"
                           :themes [:rdash]
                           :namespaces [uncomplicate.neanderthal.core
                                        uncomplicate.neanderthal.linalg
                                        uncomplicate.neanderthal.native
                                        uncomplicate.neanderthal.opencl
                                        uncomplicate.neanderthal.cuda
                                        uncomplicate.neanderthal.math
                                        uncomplicate.neanderthal.vect-math
                                        uncomplicate.neanderthal.real
                                        uncomplicate.neanderthal.auxil
                                        uncomplicate.neanderthal.random]
                           :output-path "docs/codox"}

                   ;;also replaces lein's default JVM argument TieredStopAtLevel=1
                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"]}}

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  :classifiers {:tests {:source-paths ^:replace ["test"]}}
  :source-paths ["src/clojure" "src/device"]
  :java-source-paths ["src/java"]
  :test-paths ["test"])
