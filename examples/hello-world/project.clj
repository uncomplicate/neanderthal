(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")]
  (defproject hello-world "0.2.0"
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}
    :dependencies [[org.clojure/clojure "1.7.0-RC1"]
                   [uncomplicate/neanderthal "0.2.0-SNAPSHOT"]
                   [uncomplicate/neanderthal-atlas "0.1.0" :classifier ~nar-classifier]]))
