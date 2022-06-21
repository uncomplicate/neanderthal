(defproject hello-world "0.44.0"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.3"]
                 [uncomplicate/neanderthal "0.44.0"]
                 ;;Optional. If bytedeco is not present, a system-wide MKL is used.
                 [org.bytedeco/mkl-platform-redist "2022.0-1.5.7"]]
  ;; Nvidia doesn't ship CUDA for macOS; you have to add this to your project
  :exclusions [[org.jcuda/jcuda-natives :classifier "apple-x86_64"]
               [org.jcuda/jcublas-natives :classifier "apple-x86_64"]]

  ;; If on Java 9+, you have to uncomment the following JVM option.
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"
                       "--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"
                       "--add-opens=java.base/sun.nio.ch=ALL-UNNAMED"])
