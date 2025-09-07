(defproject hello-world-aot "0.57.0-SNAPSHOT"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.2"]
                 [uncomplicate/neanderthal "0.57.0-SNAPSHOT"]]

  :profiles {:default [:default/all ~(leiningen.core.utils/get-os)]
             :default/all {:dependencies [;; optional on Linux and Windows
                                          [org.bytedeco/openblas "0.3.30-1.5.12"]]}
             :linux {:dependencies [[org.bytedeco/mkl "2025.2-1.5.12" :classifier "linux-x86_64-redist"]
                                    ;; optional, if you want GPU computing with CUDA. Beware: the jar size is 3GB!
                                    [org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.143830-1" :classifier "linux-x86_64-redist"]]}
             :windows {:dependencies [[org.bytedeco/mkl "2025.2-1.5.12" :classifier "windows-x86_64-redist"]
                                      ;; optional, if you want GPU computing with CUDA. Beware: the jar size is 3GB!
                                      [org.bytedeco/cuda "12.9-9.10-1.5.12-20250612.145546-3" :classifier "windows-x86_64-redist"]]}
             :macosx {:dependencies []}}

  ;; Wee need this for the CUDA binaries, which are not available in the Maven Central due to its huge size (3GB, vs 1GB limit)!
  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       "--enable-native-access=ALL-UNNAMED"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
