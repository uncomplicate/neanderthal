(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")

      project-version "0.1.0-SNAPSHOT"]
  (defproject uncomplicate/neanderthal "0.1.0-SNAPSHOT"
    :description "FIXME: write description"
    :url "http://example.com/FIXME"
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}
    :dependencies [[org.clojure/clojure "1.7.0-alpha3"]
                   [uncomplicate/neanderthal-atlas ~project-version]
                   [primitive-math "0.1.4"]
                   [org.apache.commons/commons-math3 "3.3"]
                   [vertigo "0.1.3"]]


    :aot [uncomplicate.neanderthal.protocols]

    :global-vars {*warn-on-reflection* true
                  *assert* false}
    :profiles {:dev {:plugins [[cider/cider-nrepl "0.8.2-SNAPSHOT"]
                               [lein-midje "3.1.3"]]
                     :dependencies [[uncomplicate/neanderthal-atlas ~project-version
                                     :classifier ~nar-classifier]
                                    [midje "1.6.3"]
                                    [criterium "0.4.3"]]}}

    :javac-options ["-target" "1.6" "-source" "1.6" "-Xlint:-options"]
    :source-paths ["src/clojure"]
    :java-source-paths ["src/java"]
    :test-paths ["test" "test/clojure"]

    ))
