(defproject benchmarks "0.24.0"
  :description "Benchmarks and comparisons between Neanderthal and other Java matrix libraries."
  :url "https://github.com/uncomplicate/neanderthal/tree/master/examples/benchmarks"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}

  :dependencies [[org.clojure/clojure "1.10.0"]
                 [criterium "0.4.4"]
                 [prismatic/hiphip "0.2.1"]
                 [net.mikera/core.matrix "0.58.0"]
                 [net.mikera/vectorz-clj "0.46.0"]
                 [clatrix/clatrix "0.5.0"]
                 [uncomplicate/neanderthal "0.24.0"]
                 [org.nd4j/nd4j-api "1.0.0-beta"]
                 #_[org.nd4j/nd4j-cuda-9.1 "1.0.0-beta"]
                 [org.nd4j/nd4j-native-platform "1.0.0-beta"]]

  ;;:repositories [["snapshots" {:url "https://oss.sonatype.org/content/repositories/snapshots"}]]
  :jvm-opts ^:replace ["-Dserver -Dclojure.compiler.direct-linking=true"
                       "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"
                       #_"-Dorg.bytedeco.javacpp.openblas.load=mkl_rt"]

  :global-vars {*warn-on-reflection* true
                *assert* false
                *unchecked-math* :warn-on-boxed
                *print-length* 16})
