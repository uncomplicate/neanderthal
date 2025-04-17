(defproject hello-world "0.54.0-SNAPSHOT"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.3"]
                 [uncomplicate/neanderthal "0.54.0-SNAPSHOT"]]

  :profiles {:default [:default/all ~(leiningen.core.utils/get-os)]
             :default/all {:dependencies [[codox-theme-rdash "0.1.2"]
                                          [midje "1.10.10"]
                                          [org.bytedeco/openblas "0.3.29-1.5.12-SNAPSHOT"]]}
             :linux {:dependencies [[org.bytedeco/openblas "0.3.29-1.5.12-SNAPSHOT" :classifier linux-x86_64]
                                    [org.bytedeco/mkl "2025.0-1.5.11" :classifier linux-x86_64-redist]
                                    [org.bytedeco/cuda "12.6-9.5-1.5.11" :classifier linux-x86_64-redist]]}
             :windows {:dependencies [[org.bytedeco/openblas "0.3.29-1.5.12-SNAPSHOT" :classifier windows-x86_64]
                                      [org.bytedeco/mkl "2025.0-1.5.11" :classifier windows-x86_64-redist]
                                      [org.bytedeco/cuda "12.6-9.5-1.5.11" :classifier windows-x86_64-redist]]}
             :macosx {:dependencies [[org.uncomplicate/neanderthal-apple "0.54.0-SNAPSHOT"]
                                     [org.bytedeco/openblas "0.3.29-1.5.12-SNAPSHOT" :classifier macosx-arm64]]}}
  ;; Wee need this for pinned openblas snapshot! If you're just using MKL (recommended on Linux and Windows) you don't need this.
  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"] ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
