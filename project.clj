;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/neanderthal "0.9.0-SNAPSHOT"
  :description "Neanderthal is a Clojure library for fast matrix and linear algebra computations."
  :url "https://github.com/uncomplicate/neanderthal"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/neanderthal"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.8.0"]
                 [uncomplicate/commons "0.2.2"]
                 [uncomplicate/fluokitten "0.5.1"]
                 [uncomplicate/clojurecl "0.6.5"]
                 [uncomplicate/neanderthal-native "0.9.0-SNAPSHOT"]
                 [org.jocl/jocl-blast "0.9.0"]
                 [org.apache.commons/commons-math3 "3.6.1"]
                 [vertigo "0.1.4"]]

  :codox {:src-dir-uri "http://github.com/uncomplicate/neanderthal/blob/master/"
          :src-linenum-anchor-prefix "L"
          :namespaces [uncomplicate.neanderthal.core
                       uncomplicate.neanderthal.native
                       uncomplicate.neanderthal.opencl
                       uncomplicate.neanderthal.math
                       uncomplicate.neanderthal.real]
          :output-path "docs/codox"}

  ;;also replaces lein's default JVM argument TieredStopAtLevel=1
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"]

  #_(:aot [uncomplicate.neanderthal.protocols
           uncomplicate.neanderthal.core
           uncomplicate.neanderthal.impl.buffer-block
           uncomplicate.neanderthal.impl.cblas])

  :profiles {:dev {:plugins [[lein-midje "3.2"]
                             [lein-codox "0.10.1"]]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[midje "1.8.3"]]}}

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :source-paths ["src/clojure" "src/opencl"]
  :java-source-paths ["src/java"]
  :test-paths ["test"])
