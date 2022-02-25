ARG CUDA_VER=11.2.0
ARG LINUX_VER=ubuntu18.04

FROM nvidia/cuda:${CUDA_VER}-devel-${LINUX_VER}

RUN apt update -y && apt-get --fix-missing upgrade -y && apt install -y git

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN bash /miniconda.sh -b -p /opt/conda


ENV GCC_VERSION=9
ENV CXX_VERSION=9
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

ARG CONDA_HOME=/opt/conda
ENV CONDA_HOME="$CONDA_HOME"

ENV CUDA_HOME="/usr/local/cuda-11.2"
ENV CUDA_PATH="$CUDA_HOME"

ARG CUDA_SHORT_VERSION
ENV CUDA_SHORT_VERSION="$CUDA_SHORT_VERSION"

ARG PARALLEL_LEVEL=4
ENV PARALLEL_LEVEL=${PARALLEL_LEVEL}
ENV PATH="$CONDA_HOME/bin:\
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\
$CUDA_HOME/bin"


SHELL ["/bin/bash", "-c"]

ENV CONDA_ENV="rapids-gpu-bdb"

COPY /rapids-gpu-bdb-cuda11.2.yml /rapids-gpu-bdb-cuda11.2.yml
RUN conda env create --name ${CONDA_ENV} --file /rapids-gpu-bdb-cuda11.2.yml
RUN conda install -n ${CONDA_ENV} -c conda-forge spacy oauth2client gspread -q -y
RUN source activate ${CONDA_ENV} && python -m spacy download en_core_web_sm


# need to configure tzdata otherwise (seemingly)
ARG DEBIAN_FRONTEND=noninteractive

RUN set -x \
 # Install dependencies
 && mkdir -p /etc/bash_completion.d \
 && apt update -y --fix-missing || true \
 && apt upgrade -y --fix-missing || true \
 && apt install -y --fix-missing software-properties-common \
 && add-apt-repository -y ppa:git-core/ppa || true \
 && apt install -y --fix-missing --no-install-recommends \
    ed vim bc nano less git wget sudo tzdata \
    apt-utils apt-transport-https \
    gcc g++ ninja-build bash-completion \
    curl libssl-dev libcurl4-openssl-dev zlib1g-dev \
    # ucx and ucx-py dependencies
    unzip automake autoconf libb2-dev libzstd-dev \
    libtool librdmacm-dev libnuma-dev \
 && bash -c "echo -e '\
deb http://archive.ubuntu.com/ubuntu/ xenial universe\n\
deb http://archive.ubuntu.com/ubuntu/ xenial-updates universe\
'" >> /etc/apt/sources.list.d/xenial.list \
 && apt update -y || true && apt install -y libibcm-dev \
 && rm /etc/apt/sources.list.d/xenial.list \
 # cleanup
 && rm -rf /var/lib/apt/lists/* /tmp/* /var/tmp/*


# Install ibverbs from MOFED
ADD https://www.mellanox.com/downloads/ofed/MLNX_OFED-5.2-2.2.0.0/MLNX_OFED_LINUX-5.2-2.2.0.0-ubuntu18.04-x86_64.tgz /MLNX_OFED_LINUX-5.2-2.2.0.0-ubuntu18.04-x86_64.tgz

RUN tar -xzf /MLNX_OFED_LINUX-5.2-2.2.0.0-ubuntu18.04-x86_64.tgz && \
 cd MLNX_OFED_LINUX-5.2-2.2.0.0-ubuntu18.04-x86_64 && \
 echo y | ./mlnxofedinstall --user-space-only --without-fw-update --without-neohost-backend\
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /MLNX_OFED_LINUX*

RUN rm -rf /MLNX_OFED_LINUX-5.2-2.2.0.0-ubuntu18.04-x86_64*

RUN /opt/conda/bin/conda install -n $CONDA_ENV -c conda-forge autoconf cython automake make libtool \
                                 pkg-config m4 \
                                 --force --no-deps -y -q

ADD creds.json /creds.json
ENV GOOGLE_SHEETS_CREDENTIALS_PATH=/creds.json

RUN conda clean --all

WORKDIR /
