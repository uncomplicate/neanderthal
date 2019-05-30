(defproject hello-world "0.24.0"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [uncomplicate/neanderthal "0.24.0"]]
  :jvm-opts ^:replace [#_"--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"])
