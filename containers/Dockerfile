FROM python:3.10-slim

ENV NVIDIA_VISIBLE_DEVICES=all
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib64

WORKDIR /DNA-Diffusion
COPY . .

RUN apt-get update && \
    apt-get install -y \
    gcc \
    build-essential \
    zlib1g-dev

RUN pip install --upgrade pip && \
    pip install -e .
