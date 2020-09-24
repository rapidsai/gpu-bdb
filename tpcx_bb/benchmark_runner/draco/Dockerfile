ARG CUDA_VERSION=10.2
ARG LINUX_VERSION=ubuntu18.04
FROM nvidia/cuda:${CUDA_VERSION}-devel-${LINUX_VERSION}

RUN apt update -y && apt upgrade -y && apt install -y git

ADD https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh /miniconda.sh
RUN bash /miniconda.sh -b -p /home/root/conda

ENV PATH=${PATH}:/home/root/conda/bin
SHELL ["/bin/bash", "-c"]
ENV CONDA_ENV=rapids-tpcx-bb

ADD https://github.com/rapidsai/tpcx-bb/raw/main/conda/rapids-tpcx-bb.yml /rapids-tpcx-bb.yml
RUN conda env create --name ${CONDA_ENV} --file /rapids-tpcx-bb.yml
RUN conda install -n ${CONDA_ENV} -c conda-forge spacy oauth2client gspread -q -y
RUN source activate ${CONDA_ENV} && python -m spacy download en_core_web_sm

ADD creds.json /creds.json
ENV GOOGLE_SHEETS_CREDENTIALS_PATH=/creds.json

RUN mkdir -p /tmp
RUN mkdir -p /root/dask-local-directory

WORKDIR /
