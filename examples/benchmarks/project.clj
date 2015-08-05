(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")]
  (defproject benchmarks "0.3.0"
    :description "Benchmarks and comparisons between Neanderthal and other Java matrix libraries."
    :url "https://github.com/uncomplicate/neanderthal/tree/master/examples/benchmarks"
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}

    :dependencies [[org.clojure/clojure "1.8.0-alpha4"]
                   [criterium "0.4.3"]
                   [primitive-math "0.1.4"]
                   [net.mikera/core.matrix "0.36.1"]
                   [net.mikera/vectorz-clj "0.31.0"]
                   [org.jblas/jblas "1.2.3"]
                   [uncomplicate/neanderthal "0.3.0-SNAPSHOT"]
                   [uncomplicate/neanderthal-atlas "0.1.0" :classifier ~nar-classifier]]

    :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                         "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"]

    :global-vars {*warn-on-reflection* true
                  *assert* false
                  *unchecked-math* :warn-on-boxed
                  *print-length* 128}))
