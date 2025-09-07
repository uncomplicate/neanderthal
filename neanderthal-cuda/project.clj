;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/neanderthal-cuda "0.57.0-SNAPSHOT"
  :description "Neanderthal's CUDA backend."
  :url "https://github.com/uncomplicate/neanderthal"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/neanderthal"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.1"]
                 [uncomplicate/commons "0.18.0-SNAPSHOT"]
                 [uncomplicate/fluokitten "0.10.0"]
                 [uncomplicate/clojurecuda "0.22.1"]
                 [org.uncomplicate/neanderthal-base "0.57.0-SNAPSHOT"]
                 [org.uncomplicate/neanderthal-opencl "0.57.0-SNAPSHOT"]]

  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]
                                 [com.github.clj-kondo/lein-clj-kondo "0.2.5"]]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :dependencies [[midje "1.10.10"]
                                      [org.uncomplicate/neanderthal-test "0.57.0-SNAPSHOT"]
                                      [org.uncomplicate/neanderthal-openblas "0.57.0-SNAPSHOT"]]
                       :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                                             "--enable-native-access=ALL-UNNAMED"]}
             :linux {:dependencies [[org.bytedeco/openblas "0.3.30-1.5.12" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.143830-1" :classifier "linux-x86_64-redist"]]}
             :windows {:dependencies [[org.bytedeco/openblas "0.3.30-1.5.12" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.145546-3" :classifier "windows-x86_64-redist"]]}}

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]

  :classifiers {:tests {:source-paths ^:replace ["test"]}}
  :source-paths ["src/clojure" "src/device"]
  :java-source-paths ["src/java"]
  :test-paths ["test"])
