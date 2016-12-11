(defproject benchmarks "0.9.0-SNAPSHOT"
  :description "Benchmarks and comparisons between Neanderthal and other Java matrix libraries."
  :url "https://github.com/uncomplicate/neanderthal/tree/master/examples/benchmarks"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}

  :dependencies [[org.clojure/clojure "1.8.0"]
                 [criterium "0.4.3"]
                 [primitive-math "0.1.4"]
                 [prismatic/hiphip "0.2.1"]
                 [net.mikera/core.matrix "0.50.0"]
                 [net.mikera/vectorz-clj "0.43.1"]
                 [org.jblas/jblas "1.2.3"]
                 [uncomplicate/fluokitten "0.5.1"]
                 [uncomplicate/neanderthal "0.9.0-SNAPSHOT"]]

  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"]

  :global-vars {*warn-on-reflection* true
                *assert* false
                *unchecked-math* :warn-on-boxed
                *print-length* 128})
