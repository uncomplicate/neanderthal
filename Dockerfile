# failing with
# Execution error (UnsatisfiedLinkError) at java.lang.ClassLoader$NativeLibrary/load0 (ClassLoader.java:-2).
#/tmp/libneanderthal-mkl-0.33.07653633467081296505.so: libmkl_rt.so: cannot open shared object file: No such file or directory

FROM clojure:lein-2.9.8-focal
RUN apt-get update && apt-get -y install git wget python3
RUN wget https://registrationcenter-download.intel.com/akdlm/irc_nas/18721/l_onemkl_p_2022.1.0.223.sh
# RUN sh ./l_onemkl_p_2022.1.0.223.sh -a --silent  --eula accept

# RUN git clone https://github.com/uncomplicate/neanderthal.git

# WORKDIR /tmp/neanderthal
# RUN git checkout e01511ff47605f2e4031d58899b303e4435d58e3

RUN lein update-in :dependencies conj "[org.bytedeco/mkl-platform-redist \"2020.3-1.5.4\"]" -- test uncomplicate.neanderthal.mkl-test
