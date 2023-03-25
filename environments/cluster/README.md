# Setup

Please note the instructions below assume you've linked or moved the contents of this directory (`environments/cluster`) to the root folder of the repository. You can accomplish this by running

```shell
ln -s ./environments/cluster/{*.sh,.env} .
```
from the root of the repository assuming you don't have any files with overlapping names (such as `.env` already stored there).

You can remove these links when you are finished with

```shell
find . -maxdepth 1 -type l -delete
```

* review copy of `.env` file
* run `chmod +x *.sh` in the directory containing these scripts to ensure they are executable
* run `install_mambaforge.sh`
* this should run `mamba init` for you, but check your `~/.bashrc` or run it yourself if you are unsure
* `source ~/.bashrc` or restart your shell
* run `mamba info` and you will see `__cuda` is absent from the `virtual packages` section on the login node
* check compute node availability with `sinfo --state=idle` vs `sinfo`
* enter an interactive job node with `slurm_interactive.sh` (should drop you into a terminal indicating a new IP address)
* run `mamba info` and you should see the `__cuda` virtual package is now found
* (LONG) install the conda environment defined in `environment.yml` with `create_conda.sh` (~10 minutes)
* exit the interactive job node with `exit`
* clone your fork of the `DNA-Diffusion` git repository to `${WORK_HOME}` and enter the directory
* ensure there are copies of both `.env` and `dnadiffusion_run.sh`
* check compute node availability with `sinfo --state=idle` vs `sinfo`
* submit batch job with `sbatch dnadiffusion_run.sh`
* check batch job STDOUT in `dnadiffusion_JOBID_TIMESTAMP.out` in the `logs` folder
* check job status and ID with `squeue --me`
* kill the job with `scancel $JOBID`