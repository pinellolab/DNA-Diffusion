## Config Structure

Current hypothetical config folder structure is as follows:

```
├── configs
    ├── callbacks
        ├── default.yaml
    ├── dataset
        ├── sequence.yaml
    ├── logger
        ├── wandb.yaml
    ├── model
        ├── unet.yaml
        ├── unet_conditional.yaml
        ├── unet_bitdiffusion.yaml
    ├── paths
        ├── default.yaml
    ├── train.yaml
```

As new items (models, datasets, etc.) are added, a corresponding config file can be included so that minimal parameter altering is needed across various experiments

## How to Run

Below contains the main training config file that can be altered to fit any training alterations that are desired.
Every parameter listed under defaults is defined within a config listed above.

<details>
<summary><b>Training config</b></summary>

```yaml
defaults:
    - model: unet_conditional
    - dataset: sequence
    - logger: wandb
    - callbacks: default

ckpt: null # path to checkpoint
seed: 42
batch_size: 32
devices: gpu
benchmark: True
ckpt_dir: # path still to be defined
accelerator: gpu
strategy: ddp
min_epochs: 5
max_epochs: 100000
gradient_clip_val: 1.0
accumulate_grad_batches: 1
log_every_n_steps: 1
check_val_every_n_epoch: 1 #for debug purposes
save_last: True
precision: 32
```

</details>

### Using hydra config in a Jupyter Notebook

Including the following at the beginning of a jupyter notebook will initialize hydra, load defined training config, and then print it.

```python
from hydra import compose, initialize
from omegaconf import OmegaConf

initialize(version_base=None, config_path="./src/configs")
cfg = compose(config_name="train")
print(OmegaConf.to_yaml(cfg))
```

When initializing hydra it is possible to override any of the default assignments.
Here is an example of overriding batch_size and seed while initializing hydra:

```python
from hydra import compose, initialize
from omegaconf import OmegaConf

initialize(version_base=None, config_path="./src/configs")
cfg = compose(overrides=["batch_size=64", "seed=1"])
print(OmegaConf.to_yaml(cfg))
```

The following link to hydra documentation provides more information on override syntax: <br/>
https://hydra.cc/docs/advanced/override_grammar/basic/ <br/>

For more information regarding hydra initialization in jupyter see the following link:
https://github.com/facebookresearch/hydra/blob/main/examples/jupyter_notebooks/compose_configs_in_notebook.ipynb

## Still To Do:

-   Alter training script to accommodate all logs we wish to track using wandb
-   Decide on default hyperparameters in train.yaml
-   Further alter config folder structure to best suit our training and testing practices
-   Define default paths for dataset within path config file so that directory can be referenced across various other configs
-   Hydra config logs currently output in src directory, creating the following folder structure ./src/outputs/YYYY-MM-DD/MM-HH-SS. If we wish to alter this it can be done in a hydra config file.
