(defproject hello-world "0.48.0"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.1"]
                 [uncomplicate/neanderthal "0.48.0o"]
                 [org.bytedeco/mkl "2024.0-1.5.10"]
                 ;; Optional. Needed for GPU engines.
                 [org.bytedeco/cuda "12.3-8.9-1.5.10"]]

  ;; :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
  ;;                      "-XX:MaxDirectMemorySize=16g" "-XX:+UseLargePages"]

  :global-vars {*warn-on-reflection* true
                *assert* false
                *unchecked-math* :warn-on-boxed
                *print-length* 16})
