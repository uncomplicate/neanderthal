(let [nar-classifier (str (System/getProperty "os.arch") "-"
                          (System/getProperty "os.name") "-gpp-jni")]
  (defproject hello-world "0.1.1"
    :license {:name "Eclipse Public License"
              :url "http://www.eclipse.org/legal/epl-v10.html"}
    :dependencies [[org.clojure/clojure "1.7.0-alpha4"]
                   [uncomplicate/neanderthal "0.1.1"]
                   [uncomplicate/neanderthal-atlas "0.1.0" :classifier ~nar-classifier]]))
