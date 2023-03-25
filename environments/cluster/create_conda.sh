#!/usr/bin/env bash

# run this after entering an interactive job with "slurm_interactive.sh"
# so that conda will pick up the correct environment for CUDA
# not available on the login nodes

mamba env create -f environments/conda/environment.yml
