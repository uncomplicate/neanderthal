;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/neanderthal-base "0.55.0-SNAPSHOT"
  :description "Neanderthal is a Clojure library for fast matrix and linear algebra computations."
  :url "https://github.com/uncomplicate/neanderthal"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/neanderthal"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.1"]
                 [uncomplicate/commons "0.17.1-SNAPSHOT"]
                 [uncomplicate/fluokitten "0.10.0"]
                 [org.uncomplicate/clojure-cpp "0.5.1-SNAPSHOT"]
                 [org.apache.commons/commons-math3 "3.6.1"]
                 [org.clojure/tools.logging "1.3.0"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [lein-codox "0.10.8"]
                             [com.github.clj-kondo/lein-clj-kondo "0.2.5"]]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[codox-theme-rdash "0.1.2"]]
                   :codox {:metadata {:doc/format :markdown}
                           :source-uri "http://github.com/uncomplicate/neanderthal/blob/master/{filepath}#L{line}"
                           :themes [:rdash]
                           :namespaces [uncomplicate.neanderthal.auxil
                                        uncomplicate.neanderthal.block
                                        uncomplicate.neanderthal.core
                                        uncomplicate.neanderthal.integer
                                        uncomplicate.neanderthal.linalg
                                        uncomplicate.neanderthal.math
                                        uncomplicate.neanderthal.native
                                        uncomplicate.neanderthal.random
                                        uncomplicate.neanderthal.real
                                        uncomplicate.neanderthal.vect-math
                                        uncomplicate.neanderthal.sparse]
                           :output-path "../docs/codox"}

                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                        "--enable-native-access=ALL-UNNAMED"]}}

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
  :source-paths ["src/clojure" "src/device"]
  :java-source-paths ["src/java"])
