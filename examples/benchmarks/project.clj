(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")]
  (defproject benchmarks "0.1.1"
    :description "Benchmarks and comparisons between Neanderthal and other Java matrix libraries."
    :url "https://github.com/uncomplicate/neanderthal/tree/master/examples/benchmarks"
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}

    :jvm-opts ^:replace ["-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"];;also replaces lein's default JVM argument TieredStopAtLevel=1

    :dependencies [[org.clojure/clojure "1.7.0-alpha5"]
                   [criterium "0.4.3"]
                   [primitive-math "0.1.4"]
                   [net.mikera/core.matrix "0.32.1"]
                   [clatrix "0.4.0"]
                   [net.mikera/vectorz-clj "0.28.0"]
                   [org.jblas/jblas "1.2.3"]
                   [uncomplicate/neanderthal "0.1.1"]
                   [uncomplicate/neanderthal-atlas "0.1.0" :classifier ~nar-classifier]]))
