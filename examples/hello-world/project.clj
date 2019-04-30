(defproject hello-world "0.23.1"
  :license {:name "Eclipse Public License"
            :url "http://www.eclipse.org/legal/epl-v10.html"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [uncomplicate/neanderthal "0.23.1"]]
  :exclusions [[org.jcuda/jcuda-natives :classifier "apple-x86_64"]
               [org.jcuda/jcublas-natives :classifier "apple-x86_64"]]
  :jvm-opts ^:replace ["-Dclojure.compiler.direct-linking=true"
                       #_"--add-opens=java.base/jdk.internal.ref=ALL-UNNAMED"])
