;;   Copyright (c) Dragan Djuric. All rights reserved.
;;   The use and distribution terms for this software are covered by the
;;   Eclipse Public License 1.0 (http://opensource.org/licenses/eclipse-1.0.php) or later
;;   which can be found in the file LICENSE at the root of this distribution.
;;   By using this software in any fashion, you are agreeing to be bound by
;;   the terms of this license.
;;   You must not remove this notice, or any other, from this software.

(defproject uncomplicate/neanderthal "0.60.2"
  :description "Convenience project to pull Ahead-Of-Time compiled neanderthal dependencies compatible with the classic Neanderthal uberproject."
  :url "https://github.com/uncomplicate/neanderthal"
  :scm {:name "git"
        :url "https://github.com/uncomplicate/neanderthal"}
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.4"]
                 [uncomplicate/commons "0.19.0"]
                 [uncomplicate/fluokitten "0.10.0"]
                 [org.uncomplicate/neanderthal-base "0.60.0"]
                 [org.uncomplicate/neanderthal-openblas "0.60.0"]
                 [org.uncomplicate/neanderthal-mkl "0.60.0"]
                 [org.uncomplicate/neanderthal-opencl "0.60.0"]
                 [org.uncomplicate/neanderthal-cuda "0.60.2"]
                 [org.uncomplicate/neanderthal-accelerate "0.60.0"]]

  :aot [uncomplicate.neanderthal.internal.cpp.structures
        uncomplicate.neanderthal.internal.cpp.factory
        uncomplicate.neanderthal.internal.cpp.mkl.factory
        uncomplicate.neanderthal.internal.cpp.openblas.factory
        uncomplicate.neanderthal.internal.cpp.cuda.structures
        uncomplicate.neanderthal.internal.cpp.cuda.factory
        uncomplicate.neanderthal.internal.cpp.accelerate.factory
        uncomplicate.neanderthal.internal.device.clblock
        uncomplicate.neanderthal.internal.device.clblast]

  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:plugins [[lein-midje "3.2.1"]
                                 [lein-codox "0.10.8"]
                                 [com.github.clj-kondo/lein-clj-kondo "0.2.5"]]
                       :global-vars {*warn-on-reflection* true
                                     *assert* false
                                     *unchecked-math* :warn-on-boxed
                                     *print-length* 128}
                       :dependencies [[midje "1.10.10"]
                                      [codox-theme-rdash "0.1.2"]]
                       :codox {:metadata {:doc/format :markdown}
                               :source-uri "http://github.com/uncomplicate/neanderthal/blob/master/{filepath}#L{line}"
                               :themes [:rdash]
                               :source-paths ["../neanderthal-base/src/clojure/"
                                              "../neanderthal-cuda/src/clojure/"
                                              "../neanderthal-opencl/src/clojure/"]
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
                                            uncomplicate.neanderthal.sparse
                                            uncomplicate.neanderthal.cuda
                                            uncomplicate.neanderthal.opencl]
                               :output-path "../docs/codox"}}
             :linux {:dependencies [[org.bytedeco/openblas "0.3.30-1.5.12" :classifier "linux-x86_64"]
                                    [org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    [org.bytedeco/cuda-redist "13.1-9.18-1.5.13-20260203.003728-10" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cublas "13.1-9.18-1.5.13-20260203.003743-10" :classifier "linux-x86_64"]]}
             :windows {:dependencies [[org.bytedeco/openblas "2025.2-1.5.12" :classifier "windows-x86_64"]
                                      [org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      [org.bytedeco/cuda-redist "13.1-9.18-1.5.13-20260203.003728-10" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cublas "13.1-9.18-1.5.13-20260203.003743-10" :classifier "windows-x86_64"]]}
             :macosx {:dependencies [[org.bytedeco/openblas "0.3.30-1.5.12" :classifier "macosx-arm64"]]}}

  :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]

  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  :source-paths ["src/clojure" "src/device"]
  :java-source-paths ["src/java"])
