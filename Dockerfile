# from https://github.com/ComputeSoftware/java-neanderthal-docker/blob/master/tools-deps/Dockerfile
FROM circleci/clojure:openjdk-8-tools-deps-1.10.0.442

USER root

RUN apt-get update
RUN apt-get install apt-transport-https -y
RUN wget -O - https://apt.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB | apt-key add -
RUN sh -c 'echo deb https://apt.repos.intel.com/mkl all main > /etc/apt/sources.list.d/intel-mkl.list'
RUN apt-get update
RUN apt-get install intel-mkl-64bit-2018.4-057 -y

ENV LD_LIBRARY_PATH /opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64_lin:/opt/intel/compilers_and_libraries_2018.5.274/linux/compiler/lib/intel64_lin

USER circleci

# from https://github.com/scicloj/docker-hub/blob/master/libpython-clj/Dockerfile
RUN echo '{:deps { uncomplicate/neanderthal {:mvn/version "0.36.0"} nrepl/nrepl {:mvn/version "0.6.0"} org.clojure/tools.deps.alpha {:mvn/version "0.6.496"} cider/cider-nrepl {:mvn/version "0.25.3"}} :aliases {:nREPL {:extra-deps {nrepl/nrepl {:mvn/version "0.6.0"}}}}}' > deps.edn &&\
    echo '{:bind "0.0.0.0" :port 8888}' > .nrepl.edn &&\
    clj -Sforce < /dev/null >&0

CMD ["clj", "-R:nREPL", "-m", "nrepl.cmdline", "--middleware", "[cider.nrepl/cider-middleware]"]
