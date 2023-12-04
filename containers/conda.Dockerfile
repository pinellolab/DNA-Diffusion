FROM mambaorg/micromamba:1.5.3-jammy-cuda-12.3.0

USER root

RUN apt-get update -yq && \
    apt-get install -yq --no-install-recommends \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR ${HOME}
USER ${MAMBA_USER}

COPY --chown=${MAMBA_USER}:${MAMBA_USER} . ${HOME}

RUN micromamba install \
    --yes \
    --channel=conda-forge \
    --name=base \
    condax
RUN /opt/conda/bin/condax install \
    --channel=conda-forge \
    --link-conflict=overwrite \
    conda-lock

ENV PATH="${PATH}:${HOME}/.local/bin"

RUN conda-lock install \
    --micromamba \
    --name=dnadiffusion \
    --extras=workflows \
    environments/conda/conda-lock.yml

ENV ENV_NAME=dnadiffusion
ARG MAMBA_DOCKERFILE_ACTIVATE=1
# If the environment is not activated,
# it is also possible to use `micromamba run`
# RUN micromamba run -n dnadiffusion \
#     pip install --no-deps -e .
RUN pip install --no-deps -e .

ARG tag
ENV FLYTE_INTERNAL_IMAGE ${tag}
