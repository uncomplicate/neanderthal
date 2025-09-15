(defproject hello-world-aot "0.58.0"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.2"]
                 [uncomplicate/neanderthal "0.58.0"]]

  ;; uncomplicate/neanderthal is AOT compiled for fast loading and developer convenience, which
  ;; might cause issues since it freezes org.clojure/core.async to the specific version (see ClojureCUDA).

  ;; FOR PRODUCTION USE, PLEASE USE org.uncomplicate/neanderthal-base AND OTHER PARTICULAR DEPENDENCIES

  :profiles {:default [:default/all ~(leiningen.core.utils/get-os)]
             :default/all {:dependencies [;; optional on Linux and Windows
                                          [org.bytedeco/openblas "0.3.30-1.5.12"]]}
             :linux {:dependencies [[org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    ;; optional, if you want GPU computing with CUDA. Beware: the size of 2 jars size is cca 800 MB.
                                    [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "linux-x86_64-redist"]
                                    [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "linux-x86_64-redist-cublas"]]}
             :windows {:dependencies [[org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      ;; optional, if you want GPU computing with CUDA. Beware: the size of 2 jars size is cca 800 MB.
                                      [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "windows-x86_64-redist"]
                                      [org.bytedeco/cuda "12.9-9.10-1.5.13-20250913.041224-9" :classifier "windows-x86_64-redist-cublas"]]}
             :macosx {:dependencies []}}

  ;; Wee need this for the CUDA binaries, which are not available in the Maven Central due to its huge size (3GB, vs 1GB limit)!
  :repositories [["maven-central-snapshots" "https://central.sonatype.com/repository/maven-snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
