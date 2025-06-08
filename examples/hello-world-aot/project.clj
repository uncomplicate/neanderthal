(defproject hello-world-aot "0.54.0-SNAPSHOT"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.12.0"]]

  :profiles {:default [:default/all ~(leiningen.core.utils/get-os)]
             :default/all {:dependencies [[codox-theme-rdash "0.1.2"]
                                          ;; optional
                                          [org.bytedeco/openblas "0.3.29-1.5.12-SNAPSHOT"]]}
             :linux {:dependencies [[uncomplicate/neanderthal "0.54.0-SNAPSHOT"]
                                    [org.bytedeco/mkl "2025.0-1.5.11" :classifier linux-x86_64-redist]
                                    ;; optional, if you want GPU computing with CUDA. Beware: the jar size is 3GB!
                                    [org.bytedeco/cuda "12.9-9.9-1.5.12-SNAPSHOT" :classifier linux-x86_64-redist]]}
             :windows {:dependencies [[uncomplicate/neanderthal "0.54.0-SNAPSHOT"]
                                      [org.bytedeco/mkl "2025.0-1.5.11" :classifier windows-x86_64-redist]
                                      ;; optional, if you want GPU computing with CUDA. Beware: the jar size is 3GB!
                                      [org.bytedeco/cuda "12.9-9.9-1.5.12-SNAPSHOT" :classifier windows-x86_64-redist]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-apple "0.54.0-SNAPSHOT"]]}}

  ;; Wee need this for the snapshots!
  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"]

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
