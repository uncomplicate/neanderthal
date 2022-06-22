# failing with
# actual result:
# clojure.lang.ExceptionInfo: LAPACK error. {:bad-argument 5, :error-code -5}
#  uncomplicate.neanderthal.internal.host.mkl.FloatSYEngine.copy(mkl.clj:2065)

FROM clojure:lein-2.9.8-focal
RUN apt-get update && apt-get -y install git wget python3 cpio
RUN wget http://registrationcenter-download.intel.com/akdlm/irc_nas/tec/16917/l_mkl_2020.4.304.tgz
RUN tar xzf l_mkl_2020.4.304.tgz
RUN cd l_mkl_2020.4.304 && ./install.sh -s silent.cfg --accept-eula
RUN git clone https://github.com/uncomplicate/neanderthal.git

WORKDIR /tmp/neanderthal
RUN git checkout e01511ff47605f2e4031d58899b303e4435d58e3
ENV LD_LIBRARY_PATH="/opt/intel/mkl/lib/intel64/"
CMD  ["lein", "update-in" ,":dependencies", "conj" ,"[org.bytedeco/mkl-platform-redist \"2020.3-1.5.4\"]" ,"--", "test" ,"uncomplicate.neanderthal.mkl-test" ]
