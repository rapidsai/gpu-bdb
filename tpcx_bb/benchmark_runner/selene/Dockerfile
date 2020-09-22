ARG CUDA_VER=11.0
ARG LINUX_VER=ubuntu18.04
ARG PYTHON_VER=3.8
ARG RAPIDS_VER=0.16

# FROM ${FROM_IMAGE}:${RAPIDS_VER}-cuda${CUDA_VER}-base-${LINUX_VER}-py${PYTHON_VER}
# FROM rapidsai/rapidsai-nightly:cuda11.0-runtime-ubuntu18.04-py3.8
FROM nvidia/cuda:${CUDA_VER}-devel-${LINUX_VER}

RUN apt update -y && apt upgrade -y && apt install -y git

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN bash /miniconda.sh -b -p /opt/conda


ENV GCC_VERSION=7
ENV CXX_VERSION=7
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++

ARG CONDA_HOME=/opt/conda
ENV CONDA_HOME="$CONDA_HOME"

ENV CUDA_HOME="/usr/local/cuda-11.0"
ENV CUDA_PATH="$CUDA_HOME"

ARG CUDA_SHORT_VERSION
ENV CUDA_SHORT_VERSION="$CUDA_SHORT_VERSION"

ARG PARALLEL_LEVEL=4
ENV PARALLEL_LEVEL=${PARALLEL_LEVEL}
ENV PATH="$CONDA_HOME/bin:\
/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:\
$CUDA_HOME/bin"


SHELL ["/bin/bash", "-c"]

ENV CONDA_ENV=rapids

COPY /rapids-tpcx-bb-cuda11.yml /rapids-tpcx-bb-cuda11.yml
RUN conda env create --name ${CONDA_ENV} --file /rapids-tpcx-bb-cuda11.yml
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
    ed vim nano less git wget sudo tzdata \
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
ADD https://www.mellanox.com/downloads/ofed/MLNX_OFED-5.0-2.1.8.0/MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64.tgz /MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64.tgz

COPY /MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64.tgz /tmp/MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64.tgz


RUN tar -xzf /tmp/MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64.tgz && \
 cd MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64 && \
 apt-get update && apt-get install -y --no-install-recommends \
    ./DEBS_UPSTREAM_LIBS/libibverbs* ./DEBS_UPSTREAM_LIBS/ibverbs-providers* \
 && rm -rf /var/lib/apt/lists/* \
 && rm -rf /tmp/MLNX_OFED_LINUX*

RUN rm -rf /MLNX_OFED_LINUX-5.0-2.1.8.0-ubuntu18.04-x86_64*

RUN /opt/conda/bin/conda install -n rapids -c conda-forge autoconf cython automake make libtool \
                                 pkg-config m4 \
                                 --force --no-deps -y -q


# Install UCX
ADD https://raw.githubusercontent.com/rapidsai/ucx-split-feedstock/master/recipe/add-page-alignment.patch /tmp/add-page-alignment.patch
ADD https://raw.githubusercontent.com/rapidsai/ucx-split-feedstock/master/recipe/ib_registration_cache.patch /tmp/ib_registration_cache.patch

RUN git clone --recurse-submodules -b v1.8.x https://github.com/openucx/ucx /tmp/ucx \
 && cd /tmp/ucx \
 && source activate rapids \
 && patch -p1 < /tmp/ib_registration_cache.patch \
 && patch -p1 < /tmp/add-page-alignment.patch \
 && ./autogen.sh \
 && ./configure \
    --prefix="${CONDA_PREFIX}" \
    --with-sysroot \
    --enable-cma \
    --enable-mt \
    --enable-numa \
    --with-gnu-ld \
    --with-cm \
    --with-rdmacm \
    --with-verbs \
    --with-cuda="${CUDA_HOME}" \
 && make -j${PARALLEL_LEVEL} \
 && make install

# Install UCX-py
RUN set -x \
 && git clone --recurse-submodules -j${PARALLEL_LEVEL} -b branch-0.16 https://github.com/rapidsai/ucx-py.git /tmp/ucx-py \
 && source activate rapids \
 && cd /tmp/ucx-py \
 && python setup.py build_ext --inplace \
 && python -m pip install . --no-deps --no-cache-dir -vvvv \
 && rm -rf /tmp/ucx-py

# use login shell when exec'ing into the container
# SHELL ["/bin/bash", "-l"]

ADD creds.json /creds.json
ENV GOOGLE_SHEETS_CREDENTIALS_PATH=/creds.json

RUN conda clean --all

WORKDIR /


