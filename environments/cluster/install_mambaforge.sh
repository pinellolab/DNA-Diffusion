#!/usr/bin/env bash

# set -euxo pipefail
set -x

# https://askubuntu.com/a/1389915/21876
[ ! -f .env ] || source .env

# https://unix.stackexchange.com/q/569988/9185
export | cut -d" " -f3-

echo "$PWD"

# https://github.com/conda-forge/miniforge#install
wget "https://github.com/conda-forge/miniforge/releases/latest/download/Mambaforge-$(uname)-$(uname -m).sh"

bash Mambaforge-$(uname)-$(uname -m).sh -f -b -p ${WORK_HOME}/${CONDATYPE}

${WORK_HOME}/${CONDATYPE}/condabin/mamba init