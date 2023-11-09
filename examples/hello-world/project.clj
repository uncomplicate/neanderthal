(defproject hello-world "0.48.0-SNAPSHOT"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [uncomplicate/neanderthal "0.48.0-SNAPSHOT"]
                 ;;Optional. If bytedeco mkl is not present, a system-wide MKL is used.
                 [org.bytedeco/mkl "2023.1-1.5.10-SNAPSHOT" :classifier linux-x86_64-redist]
                 [org.bytedeco/cuda "12.1-8.9-1.5.10-SNAPSHOT" :classifier linux-x86_64-redist]]

  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"]

  :global-vars {*warn-on-reflection* true
                *assert* false
                *unchecked-math* :warn-on-boxed
                *print-length* 16})
