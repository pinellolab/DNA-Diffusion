#!/usr/bin/env bash

srun --account openbioml --partition=g40 --gpus=1 --cpus-per-gpu=12 --job-name=obmldd --pty bash -i
