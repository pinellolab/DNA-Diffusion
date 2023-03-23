# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:22.11.1-4
WORKDIR /DNA-Diffusion
COPY . .

RUN mamba init
RUN mamba info && \
    mamba install python=3.10 --yes
RUN pip install -e .
