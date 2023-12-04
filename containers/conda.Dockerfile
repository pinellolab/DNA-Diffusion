FROM mambaorg/micromamba:1.5.3-jammy-cuda-12.3.0

WORKDIR /root
ENV HOME /root
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /root

RUN micromamba create \
    --name=dnadiffusion \
    --category=main \
    --category=workflows \
    --file environment/conda/conda-lock.yml
RUN conda activate dnadiffusion
RUN pip install --no-deps -e .

# Set additional ARG and ENV as needed
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
