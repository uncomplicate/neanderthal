(defproject hello-world "0.37.0"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [uncomplicate/neanderthal "0.37.0"]
                 ;;Optional. If bytedeco is not present, a system-wide MKL is used.
                 [org.bytedeco/mkl-platform-redist "2020.3-1.5.4"]]
  ;; If on Java 9+, you have to uncomment the following JVM option.
  :jvm-opts ^:replace [#_"--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"])
