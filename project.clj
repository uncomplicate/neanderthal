(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")
      atlas-version "0.1.0"]
  (defproject uncomplicate/neanderthal "0.5.0-SNAPSHOT"
    :description "Neanderthal is a Clojure library for fast matrix and linear algebra computations."
    :url "https://github.com/uncomplicate/neanderthal"
    :scm {:name "git"
          :url "https://github.com/uncomplicate/neanderthal"}
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}
    :dependencies [[org.clojure/clojure "1.8.0"]
                   [uncomplicate/fluokitten "0.4.0-SNAPSHOT"]
                   [uncomplicate/clojurecl "0.4.0-SNAPSHOT"]
                   [uncomplicate/neanderthal-atlas ~atlas-version]
                   [org.apache.commons/commons-math3 "3.6"]
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
             uncomplicate.neanderthal.impl.buffer-block
             uncomplicate.neanderthal.impl.cblas
             uncomplicate.neanderthal.opencl.clblock
             uncomplicate.neanderthal.opencl.amd-gcn])

    :profiles {:dev {:plugins [[lein-midje "3.1.3"]
                               [lein-codox "0.9.4"]]
                     :global-vars {*warn-on-reflection* true
                                   *assert* false
                                   *unchecked-math* :warn-on-boxed
                                   *print-length* 128}
                     :dependencies [[uncomplicate/neanderthal-atlas ~atlas-version
                                     :classifier ~nar-classifier]
                                    [midje "1.8.3"]
                                    [criterium "0.4.4"]]}}

    :javac-options ["-target" "1.8" "-source" "1.8" "-Xlint:-options"]
    :source-paths ["src/clojure" "src/opencl"]
    :java-source-paths ["src/java"]
    :test-paths ["test"]))
