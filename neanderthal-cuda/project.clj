;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject org.uncomplicate/neanderthal-cuda "0.54.0-SNAPSHOT"
  :description "Neanderthal's CUDA backend."
  :url "https://github.com/uncomplicate/neanderthal"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/neanderthal"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.0"]
                 [uncomplicate/commons "0.16.1"]
                 [uncomplicate/fluokitten "0.10.0"]
                 [uncomplicate/clojurecuda "0.21.0"]
                 [org.uncomplicate/neanderthal-base "0.54.0-SNAPSHOT"]
                 [org.uncomplicate/neanderthal-opencl "0.54.0-SNAPSHOT"]]

  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]
                                 [com.github.clj-kondo/lein-clj-kondo "0.2.5"]]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :dependencies [[codox-theme-rdash "0.1.2"]
                                      [midje "1.10.10"]
                                      [org.uncomplicate/neanderthal-test "0.54.0-SNAPSHOT"]
                                      [org.uncomplicate/neanderthal-openblas "0.54.0-SNAPSHOT"]
                                      [org.bytedeco/openblas "0.3.29-1.5.12-SNAPSHOT"]]
                       :codox {:metadata {:doc/format :markdown}
                               :source-uri "http://github.com/uncomplicate/neanderthal/blob/master/{filepath}#L{line}"
                               :themes [:rdash]
                               :namespaces [uncomplicate.neanderthal.cuda]
                               :output-path "../docs/codox"}
                       :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"]}
             :linux {:dependencies [[org.bytedeco/openblas "0.3.29-1.5.12-SNAPSHOT" :classifier linux-x86_64]
                                    [org.bytedeco/cuda "12.6-9.5-1.5.11" :classifier linux-x86_64-redist]]}
             :windows {:dependencies [[org.bytedeco/openblas "0.3.29-1.5.12-SNAPSHOT" :classifier windows-x86_64]
                                      [org.bytedeco/cuda "12.6-9.5-1.5.11" :classifier windows-x86_64-redist]]}}

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]

  :classifiers {:tests {:source-paths ^:replace ["test"]}}
  :source-paths ["src/clojure" "src/device"]
  :java-source-paths ["src/java"]
  :test-paths ["test"])
