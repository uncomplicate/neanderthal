FROM docker.io/ubuntu:focal
RUN apt-get update && apt-get -y install --reinstall ca-certificates && update-ca-certificates
RUN apt-get update && apt-get -y install
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y openjdk-11-jdk curl rlwrap libssl-dev build-essential zlib1g-dev  libncurses5-dev \
    libgdbm-dev libnss3-dev  libreadline-dev libffi-dev libbz2-dev  automake-1.15 git liblzma-dev wget git clinfo vim  \
    leiningen software-properties-common vim

# mkl . This might ask questions '??' Important ??
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y intel-mkl

# CUDA 11.4 via official installer
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
RUN mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
RUN dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.1-470.57.02-1_amd64.deb
RUN apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
RUN apt-get update
RUN DEBIAN_FRONTEND=noninteractive apt-get -y install cuda



#cudnn via official installer
ENV cudnn_version=8.2.4.15
ENV cuda_version=cuda11.4
ENV OS=ubuntu2004
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/cuda-${OS}.pin
RUN mv cuda-${OS}.pin /etc/apt/preferences.d/cuda-repository-pin-600
RUN apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-keys A4B469963BF863CC
RUN add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/${OS}/x86_64/ /"
RUN apt-get update
RUN apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
RUN apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}

#opencl
RUN apt-get install -y libnvidia-compute-515 nvidia-opencl-dev intel-opencl-icd
# clone neanderthal, temporay stuff just to ease to run the tests

RUN curl -O https://download.clojure.org/install/linux-install-1.11.1.1149.sh
RUN chmod +x linux-install-1.11.1.1149.sh
RUN ./linux-install-1.11.1.1149.sh

COPY deps.edn.minimal /tmp/deps.edn
WORKDIR /tmp

# from know onwards neandertthal is setup with MKL, Cuda, OpenCL
CMD ["clj" , "-M:nREPL",  "-m" , "nrepl.cmdline"]
