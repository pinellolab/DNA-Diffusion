# syntax=docker/dockerfile:1
FROM condaforge/mambaforge:23.1.0-4

ARG CONDA_OVERRIDE_CUDA=12.1
ENV NVIDIA_VISIBLE_DEVICES=all
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64

WORKDIR /DNA-Diffusion
COPY . .

RUN mamba env update -n base -f environments/conda/environment.yml && \
    pip install --no-deps -e .
