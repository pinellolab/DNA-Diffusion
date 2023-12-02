# Set args for Python and CUDA versions
ARG CUDA_VERSION=12.3.0
# TODO: Cache can cause failure to set ENV from ARG.
#       See hardcoded ENV below.
# ARG PYTHON_VERSION=3.10

# For GPU-enabled images, use nvidia/cuda as the base
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04
# See Dockerfile sources at
#
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.3.0/ubuntu2204/base/Dockerfile
# https://gitlab.com/nvidia/container-images/cuda/-/blob/master/dist/12.3.0/ubuntu2204/runtime/Dockerfile
#
# If GPUs are available and the image is not derived from nvidia/cuda the
# following ENVs are required:
#
# ENV PATH /usr/local/nvidia/bin:/usr/local/cuda/bin:${PATH}
# ENV LD_LIBRARY_PATH /usr/local/nvidia/lib:/usr/local/nvidia/lib64
# ENV NVIDIA_VISIBLE_DEVICES all
# ENV NVIDIA_DRIVER_CAPABILITIES compute,utility

WORKDIR /root
ENV HOME /root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

# Install pyenv
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -yq \
    && apt-get install -yq \
    build-essential \
    curl \
    git \
    tree \
    make \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl https://pyenv.run | bash

ENV VENV /opt/venv
ENV PYTHONPATH /root
ENV PYENV_ROOT="/root/.pyenv"
ENV PATH="${PYENV_ROOT}/shims:${PYENV_ROOT}/bin:${PATH}"
# TODO: Cache can lead to failure to set ENV from ARG
# ENV PYTHON_VERSION=${PYTHON_VERSION}
ENV PYTHON_VERSION=3.10

RUN eval "$(pyenv init -)" && \
    /root/.pyenv/bin/pyenv install --skip-existing ${PYTHON_VERSION} && \
    /root/.pyenv/bin/pyenv global ${PYTHON_VERSION}

# Setup venv for package installation
RUN python -m venv ${VENV}
ENV PATH="${VENV}/bin:$PATH"

COPY . /root

# Use local editable installation for development
# add package extras if needed for debugging e.g. `-e .[dev,docs]`
RUN pip install --upgrade pip && \
    pip install -e .[workflows]

# Install pinned version from PyPI instead for production
# RUN pip install dnadiffusion==0.1.0

ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
