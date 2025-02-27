(defproject hello-world "0.53.2"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.3"]
                 [uncomplicate/neanderthal "0.53.2"]
                 [org.bytedeco/mkl "2025.0-1.5.11" :classifier linux-x86_64-redist]
                 [org.bytedeco/cuda "12.6-9.5-1.5.11" :classifier linux-x86_64-redist]
                 ;; On macOS, remove the dependencies to MKL and CUDA!
                 ;; On windows, replace the last dependencies lines with:
                 ;;[org.bytedeco/mkl "2025.0-1.5.11" :classifier windows-x86_64-redist]
                 ;;[org.bytedeco/cuda "12.6-9.5-1.5.11" :classifier windows-x86_64-redist]
                 ;; We use OpenBLAS snapshot until JavaCPP 1.5.12 is released.
                 ;; If you're just using MKL (recommended on Linux and Windows), you don't need this, but it's supported.
                 ;; On macOS, you need it.
                 [org.bytedeco/openblas "0.3.28-1.5.12-20250223.142442-74"]]

  ;; Wee need this for pinned openblas snapshot! If you're just using MKL (recommended on Linux and Windows) you don't need this.
  :repositories [["snapshots" "https://oss.sonatype.org/content/repositories/snapshots"]]

  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"] ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
