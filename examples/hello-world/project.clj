(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")]
  (defproject hello-world "0.5.0-SNAPSHOT"
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}
    :dependencies [[org.clojure/clojure "1.8.0"]
                   [uncomplicate/neanderthal "0.5.0-SNAPSHOT"]
                   [uncomplicate/neanderthal-atlas "0.2.1-SNAPSHOT"]]))
