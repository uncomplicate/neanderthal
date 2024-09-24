(defproject benchmarks "0.48.0-SNAPSHOT"
  :description "Benchmarks and comparisons between Neanderthal and other Java matrix libraries."
  :url "https://github.com/uncomplicate/neanderthal/tree/master/examples/benchmarks"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}

  :dependencies [[org.clojure/clojure "1.11.1"]
                 [uncomplicate/neanderthal "0.48.0-SNAPSHOT"]
                 [org.bytedeco/mkl "2023.1-1.5.10-SNAPSHOT" :classifier linux-x86_64-redist]
                 [org.bytedeco/cuda "12.1-8.9-1.5.10-SNAPSHOT" :classifier linux-x86_64-redist]
                 [criterium "0.4.6"]
                 [prismatic/hiphip "0.2.1"]
                 [net.mikera/core.matrix "0.63.0"]
                 [net.mikera/vectorz-clj "0.48.0"]
                 [clatrix/clatrix "0.5.0"]
                 [org.nd4j/nd4j-api "1.0.0-M2.1"]
                 #_[org.nd4j/nd4j-cuda-9.1 "1.0.0-beta"]
                 #_[org.nd4j/nd4j-native-platform "1.0.0-beta"]]

  ;;:repositories [["snapshots" {:url "https://oss.sonatype.org/content/repositories/snapshots"}]]
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"
                       #_"-Dorg.bytedeco.javacpp.openblas.load=mkl_rt"]

  :global-vars {*warn-on-reflection* true
                *assert* false
                *unchecked-math* :warn-on-boxed
                *print-length* 16})
