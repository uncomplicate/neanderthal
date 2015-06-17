(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")

      atlas-version "0.1.0"]
  (defproject uncomplicate/neanderthal "0.2.0"
    :description "Neanderthal is a Clojure library for fast matrix and linear algebra computations."
    :url "https://github.com/uncomplicate/neanderthal"
    :scm {:name "git"
          :url "https://github.com/uncomplicate/neanderthal"}
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}
    :dependencies [[org.clojure/clojure "1.7.0-RC1"]
                   [uncomplicate/neanderthal-atlas ~atlas-version]
                   [org.apache.commons/commons-math3 "3.3"]
                   [vertigo "0.1.3"]]

    :codox {:src-dir-uri "http://github.com/uncomplicate/neanderthal/blob/master/"
            :src-linenum-anchor-prefix "L"
            :exclude [uncomplicate.neanderthal.cblas
                      uncomplicate.neanderthal.protocols]
            :output-dir "docs/codox"}

    :jvm-opts ^:replace ["-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"];;also replaces lein's default JVM argument TieredStopAtLevel=1
    :aot [uncomplicate.neanderthal.protocols]

    :profiles {:dev {:plugins [[lein-midje "3.1.3"]
                               [bilus/lein-marginalia "0.8.8"]
                               [codox "0.8.12"]]
                     :global-vars {*warn-on-reflection* true
                                   *assert* true
                                   *unchecked-math* :warn-on-boxed
                                   *print-length* 128}
                     :dependencies [[uncomplicate/neanderthal-atlas ~atlas-version
                                     :classifier ~nar-classifier]
                                    [midje "1.7.0-beta1"]
                                    [criterium "0.4.3"]]}}

    :javac-options ["-target" "1.6" "-source" "1.6" "-Xlint:-options"]
    :source-paths ["src/clojure"]
    :java-source-paths ["src/java"]
    :test-paths ["test" "test/clojure"]))
