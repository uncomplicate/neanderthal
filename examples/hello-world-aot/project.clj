(defproject hello-world-aot "0.61.0"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.4"]
                 [uncomplicate/neanderthal "0.61.0"]]

  ;; uncomplicate/neanderthal is AOT compiled for fast loading and developer convenience, which
  ;; might cause issues since it freezes org.clojure/core.async to the specific version (see ClojureCUDA).

  ;; FOR PRODUCTION USE, PLEASE USE org.uncomplicate/neanderthal-base AND OTHER PARTICULAR DEPENDENCIES

  :profiles {:dev [:dev/all ~(leiningen.core.utils/get-os)]
             :dev/all {:dependencies [;; optional on Linux and Windows, mandatory on MacOS
                                      [org.bytedeco/openblas "0.3.31-1.5.13"]]}
             :linux {:dependencies [[org.bytedeco/mkl "2025.3-1.5.13" :classifier "linux-x86_64-redist"]
                                    ;; optional, if you want GPU computing with CUDA. Beware: the size of these 2 jars is cca 800 MB.
                                    [org.bytedeco/cuda-redist "13.1-9.19-1.5.13" :classifier "linux-x86_64"]
                                    [org.bytedeco/cuda-redist-cublas "13.1-9.19-1.5.13" :classifier "linux-x86_64"]]}
             :windows {:dependencies [[org.bytedeco/mkl "2025.3-1.5.13" :classifier "windows-x86_64-redist"]
                                      ;; optional, if you want GPU computing with CUDA. Beware: the size of these 2 jars is cca 800 MB.
                                      [org.bytedeco/cuda-redist "13.1-9.19-1.5.13" :classifier "windows-x86_64"]
                                      [org.bytedeco/cuda-redist-cublas "13.1-9.19-1.5.13" :classifier "windows-x86_64"]]}
             :macosx {:dependencies []}}

  ;; We sometimes need this for the snapshot binaries of the upstream libraries.
  ;; :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
