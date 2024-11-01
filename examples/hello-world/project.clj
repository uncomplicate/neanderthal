(defproject hello-world "0.50.0"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.11.3"]
                 [uncomplicate/neanderthal "0.50.0"]
                 [org.bytedeco/mkl "2024.0-1.5.10" :classifier linux-x86_64-redist]
                 [org.bytedeco/cuda "12.3-8.9-1.5.10" :classifier linux-x86_64-redist]
                 ;; On windows, replace the last two lines with:
                 ;;[org.bytedeco/mkl "2024.0-1.5.10" :classifier windows-x86_64-redist]
                 ;;[org.bytedeco/cuda "12.3-8.9-1.5.10" :classifier windows-x86_64-redist]
                 ]

  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"] ;; We need direct linking for properly resolving types in heavy macros and avoiding reflection warnings!

  ;; :global-vars {*warn-on-reflection* true
  ;;               *assert* false
  ;;               *unchecked-math* :warn-on-boxed
  ;;               *print-length* 16}
  )
