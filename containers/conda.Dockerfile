# Use the specified micromamba image as the base
FROM mambaorg/micromamba:1.5.3-jammy-cuda-12.3.0

USER root

RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /home/mambauser
ENV LANG C.UTF-8
ENV LC_ALL C.UTF-8

USER $MAMBA_USER
RUN echo $HOME

COPY --chown=$MAMBA_USER:$MAMBA_USER . /home/mambauser

# Create the Conda environment using micromamba
RUN micromamba create \
    --yes \
    --name=dnadiffusion \
    --category=main \
    --category=workflows \
    --file environment/conda/conda-lock.yml && \
    micromamba clean --all --yes

ARG MAMBA_DOCKERFILE_ACTIVATE=1
RUN pip install --no-deps -e .

# Set additional ARG and ENV as needed
ARG tag
ENV FLYTE_INTERNAL_IMAGE $tag
