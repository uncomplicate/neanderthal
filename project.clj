(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")

      project-version "0.1.0-SNAPSHOT"]
  (defproject uncomplicate/neanderthal "0.1.0-SNAPSHOT"
    :description "FIXME: write description"
    :url "https://github.com/uncomplicate/neanderthal"
    :scm {:name "git"
          :url "https://github.com/uncomplicate/neanderthal"}
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}
    :dependencies [[org.clojure/clojure "1.7.0-alpha4"]
                   [uncomplicate/neanderthal-atlas ~project-version]
                   [primitive-math "0.1.4"]
                   [org.apache.commons/commons-math3 "3.3"]
                   [vertigo "0.1.3"]]

    :codox {:src-dir-uri "http://github.com/uncomplicate/neanderthal/blob/master"
            :src-linenum-anchor-prefix "L"
            :exclude [uncomplicate.neanderthal.cblas
                      uncomplicate.neanderthal.protocols]
            :output-dir "docs/codox"}

    :aot [uncomplicate.neanderthal.protocols]

    :profiles {:dev {:plugins [[cider/cider-nrepl "0.8.2-SNAPSHOT"]
                               [lein-midje "3.1.3"]
                               [lein-marginalia "0.8.0"]
                               [codox "0.8.10"]]
                     :global-vars {*warn-on-reflection* true
                                   *assert* false}
                     :dependencies [[uncomplicate/neanderthal-atlas ~project-version
                                     :classifier ~nar-classifier]
                                    [midje "1.6.3"]
                                    [criterium "0.4.3"]]}}

    :javac-options ["-target" "1.6" "-source" "1.6" "-Xlint:-options"]
    :source-paths ["src/clojure"]
    :java-source-paths ["src/java"]
    :test-paths ["test" "test/clojure"]))
