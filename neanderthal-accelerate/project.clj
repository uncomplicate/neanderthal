;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/neanderthal-accelerate "0.54.0"
  :description "Neanderthal's Apple Accelerate engine."
  :url "https://github.com/uncomplicate/neanderthal/neanderthal-openblas"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/neanderthal"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.1"]
                 [uncomplicate/commons "0.17.0"]
                 [uncomplicate/fluokitten "0.10.0"]
                 [org.uncomplicate/neanderthal-base "0.54.0"]
                 [org.uncomplicate/neanderthal-openblas "0.54.0"]
                 [org.uncomplicate/accelerate-platform "0.1.0-1.5.12"]]

  :profiles {:dev {:plugins [[lein-midje "3.2.1"]
                             [com.github.clj-kondo/lein-clj-kondo "0.2.5"]]
                   :global-vars {*warn-on-reflection* true
                                 *assert* false
                                 *unchecked-math* :warn-on-boxed
                                 *print-length* 128}
                   :dependencies [[codox-theme-rdash "0.1.2"]
                                  [midje "1.10.10"]
                                  [org.uncomplicate/neanderthal-test "0.54.0"]
                                  [org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]]
                   :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                        "--enable-native-access=ALL-UNNAMED"]}}

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"])
