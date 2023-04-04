**Table of Contents**

-   [DNA-Diffusion](#project-name-dnadiffusion)
    -   [Introduction](#introduction)
    -   [Hypothetical usage](#usage)
    -   [Functional Requirements](#functional-requirements)
        -   [Architecture and Design](#architecture-and-design)
        -   [Developer](#developer)
        -   [User](#user)   
    -   [Non-functional Requirements](#non-functional-requirements)
    -   [Data Model](#data-model)
    -   [External interfaces](#external-interfaces)
    -   [Project structure](#structure)
    -   [Testing](#testing)
    -   [Deployment and Maintenance](#deployment-and-maintenance)

## Introduction

As already outlined in README here we intend to learn and apply the rules underlying regulatory element sequences using Stable Diffusion.

The architecture of DNA-Diffusion intends to loosely adhere to the principles of Test-Driven Design (TDD),

Here are the main principles we strive to follow:

1. Write tests first: In TDD, you write a failing test before writing any production code. The test should be small, specific, and test only one aspect of the code.

2. Write the simplest code that passes the test: Once you've written the test, write the production code that will make the test pass. The code should be the simplest possible code that satisfies the test.

3. Refactor the code: Once the test has passed, you should refactor the code to improve its quality and maintainability. Refactoring means making changes to the code without changing its behavior.

4. Repeat the process: Once you've refactored the code, you should write another test and repeat the process.

5. Test everything that could possibly break: In TDD, you should write tests for all of the functionality that could potentially break. This includes boundary conditions, edge cases, and error conditions.

6. Use test automation: TDD relies on automated tests to verify that the code works as expected. Writing tests manually can be time-consuming and error-prone, so you should use test automation tools to write and run your tests.

7. Keep the feedback loop short: TDD is based on a short feedback loop, where you write a test, run it, and get immediate feedback on whether the code works as expected. This short feedback loop helps you catch errors early and makes it easier to debug problems.

## Hypothetical usage

First we depict a hypothetical usage example for the training:

```python
from dnadiffusion.configs import LightningTrainer, sample
Config = make_config(
    hydra_defaults=[
        "_self_",
        {"data": "LoadingData"},
        {"model": "Unet"},
    ],
    data=MISSING,
    model=MISSING,
    trainer=LightningTrainer,
    sample=sample,
    # Constants
    data_dir="dna_diffusion/data",
    random_seed=42,
    ckpt_path=None,
)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
def train(config):
    data = instantiate(config.data)
    sample = instantiate(config.sample, data_module=data)
    model = instantiate(config.model)
    trainer = instantiate(config.trainer)
    # Adding custom callbacks
    trainer.callbacks.append(sample)
    trainer.fit(model, data)
    return model
@hydra.main(config_path=None, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    return train(cfg)
```

Another usage example depicting the sampling part:

```python
Config = make_config(
    hydra_defaults=[
        "_self_",
        {"data": "LoadingData"},
        {"model": "Unet"},
    ],
    model=MISSING,
    # Constants
    random_seed=42,
    ckpt_path=None,
)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
def sample(config):
    model = instantiate(ckpt_path)
    dna_samples=model.sample(config.args)
    return dna_samples
@hydra.main(config_path=None, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    return sample(cfg)
```

## Functional Requirements

### Architecture and Design

TODO finish after final refactoring is in place.

Data
 - LoadingData: Handles loading data and generating fastas and motifs.
 - LoadingDataModule: Loading the data as adjusted PytorchLightningDataModule format.
Metrics
 - sampling_metrics.py: This file contains functions (metrics) that are used to judge the sampling process, things like KL divergence etc.
 - validation_metrics.py: This file contains functions (metrics) that are used as to judge the generated sequences themselves. Things like Enformer and BPNet will be used.
Model
 - UnetDiffusion: Defines the UNET it models with all bells and whistles (scheduler etc.)
 - Unet: Defines the bare bones UNET model.

### Developer

The library is packaged with hatch. Developer usage is documented in `README.md`.

Abstracted commands to package this software and publish it can be found in Makefile and used with make commands.

### User

Here we present a short hypothetical example based on conditional generation, i.e. with text input:

"A sequence that will correspond to open (or closed) chromatin in cell type X"

```python
TEXT_PROMPT = "A sequence that will correspond to open (or closed) chromatin in cell type X"
Config = make_config(
    hydra_defaults=[
        "_self_",
        {"data": "LoadingData"},
        {"model": "Unet"},
    ],
    model=MISSING,
    # Constants
    random_seed=42,
    ckpt_path=None,
)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)
def sample(config):
    model = instantiate(ckpt_path)
    # conditional sampling based on the text prompt
    dna_samples=model.sample(config.args, TEXT_PROMPT)
    return dna_samples
@hydra.main(config_path=None, config_name="config", version_base="1.3")
def main(cfg: DictConfig):
    return sample(cfg)
```

## Non-functional Requirements

1. Ethics and Security: The system should protect sensitive data and prevent unauthorized access or tampering. It should also be able to detect and respond to security threats or attacks.

2. Usability: The system should be easy to use and intuitive for users, including data scientists, developers, and end-users. It should have a user-friendly interface and provide clear feedback on errors and warnings.

3. Maintainability: The system should be easy to maintain and update over time, including managing data, updating the model, and fixing bugs. It should also be compatible with existing infrastructure and tools.

4. Performance: The system should be able to process data and generate predictions in a timely manner, meeting specific performance requirements or benchmarks. This includes factors like throughput, latency, and response time.

5. Scalability: The system should be able to handle large volumes of data and users, and be able to scale up or down as needed. This includes considerations such as system capacity, access to the HPC cluster and data storage.


## Data Model

We take the data presented in the following part and one-hot encode it before passing it to the network (UNET and/or UNET plus VQ_VAE).

More concretely for every sequence (200 BP in Wouters dataset) we take the nucleotides and one-hot encode them. Meaning if the input is an array of length 200, after processing its (200x4), for 4 nucleotides.

As stated in the Readme: Chromatin (DNA + associated proteins) that is actively used for the regulation of genes (i.e. "regulatory elements") is typically accessible to DNA-binding proteins such as transcription factors ([review](https://www.nature.com/articles/s41576-018-0089-8), [relevant paper](https://www.nature.com/articles/nature11232)).
Through the use of a technique called [DNase-seq](https://en.wikipedia.org/wiki/DNase-Seq), we've measured which parts of the genome are accessible across 733 human biosamples encompassing 438 cell and tissue types and states, resulting in more than 3.5 million DNase Hypersensitive Sites (DHSs).
Using Non-Negative Matrix Factorization, we've summarized these data into 16 _components_, each corresponding to a different cellular context (e.g. 'cardiac', 'neural', 'lymphoid').

For the efforts described in this proposal, and as part of an earlier [ongoing project](https://www.meuleman.org/research/synthseqs/) in the research group of Wouter Meuleman,
we've put together smaller subsets of these data that can be used to train models to generate synthetic sequences for each NMF component.

Please find these data, along with a data dictionary, [here](https://www.meuleman.org/research/synthseqs/#material).

## External interfaces

Hosted and exposed on Hugging face.

## Project structure

```
DNA-Diffusion
├─ .editorconfig
├─ .git
│  ├─ COMMIT_EDITMSG
│  ├─ config
│  ├─ description
│  ├─ HEAD
│  ├─ hooks
│  │  ├─ applypatch-msg.sample
│  │  ├─ commit-msg.sample
│  │  ├─ fsmonitor-watchman.sample
│  │  ├─ post-update.sample
│  │  ├─ pre-applypatch.sample
│  │  ├─ pre-commit.sample
│  │  ├─ pre-merge-commit.sample
│  │  ├─ pre-push.sample
│  │  ├─ pre-rebase.sample
│  │  ├─ pre-receive.sample
│  │  ├─ prepare-commit-msg.sample
│  │  ├─ push-to-checkout.sample
│  │  └─ update.sample
│  ├─ index
│  ├─ info
│  │  └─ exclude
│  ├─ objects
│  │  ├─ 18
│  │  │  └─ 0163ba3f5f8e87b6db7dbc726760ff150f65d3
│  │  ├─ 1d
│  │  │  └─ e21654cad20529e8ae0f78c449367f6c08b940
│  │  ├─ 28
│  │  │  └─ 3d132c7c47946296bcb68f3f4578c324f6d935
│  │  ├─ 44
│  │  │  └─ f71d2877c303020b91f66ec429369fa222c8fe
│  │  ├─ 4a
│  │  │  └─ da63485eaba6f60b037c658274beb189a17562
│  │  ├─ 56
│  │  │  └─ 99d40e6e22ddc2b2c5f08230ebb202dae0a486
│  │  ├─ 78
│  │  │  └─ 0ab42447c09a32c5f359fd1b6112ecb8db3fec
│  │  ├─ 8f
│  │  │  └─ b747c9644f561fe517fc11c818c987b4426b6c
│  │  ├─ 91
│  │  │  └─ 74b3b17e77aaf08911f4f90e6a19bc4cf32968
│  │  ├─ 99
│  │  │  └─ 2cea1584ba0719fa354fcf7ed27826a845e4e9
│  │  ├─ e0
│  │  │  ├─ 0dc52b076c773cd423e9c77f8f058afd87303c
│  │  │  └─ d6d62581e9e302b37f22989ac2f8d3eae97ad6
│  │  ├─ info
│  │  └─ pack
│  │     ├─ pack-50f2563450901276069c6c0124bdf552fb248dcb.idx
│  │     └─ pack-50f2563450901276069c6c0124bdf552fb248dcb.pack
│  ├─ packed-refs
│  └─ refs
│     ├─ heads
│     │  ├─ dna_diff_specification
│     │  ├─ dna_diff_specifications
│     │  └─ main
│     ├─ remotes
│     │  └─ origin
│     │     ├─ dna_diff_specifications
│     │     └─ HEAD
│     └─ tags
├─ .github
│  ├─ labels.yml
│  ├─ PULL_REQUEST_TEMPLATE.md
│  ├─ release-drafter.yml
│  └─ workflows
│     ├─ build.yml
│     ├─ docker.yml
│     ├─ documentation.yml
│     ├─ inactive-issues-prs.yml
│     ├─ labeler.yml
│     ├─ release.yml
│     └─ test-release.yml
├─ .gitignore
├─ CITATION.cff
├─ CODE_OF_CONDUCT.md
├─ dockerfiles
│  └─ Dockerfile
├─ docs
│  ├─ contributors.md
│  ├─ images
│  │  ├─ diff_first.gif
│  │  └─ diff_first_lossy.gif
│  ├─ index.md
│  ├─ reference
│  │  └─ dnadiffusion.md
│  └─ specification.md
├─ environments
│  ├─ cluster
│  │  ├─ create_conda.sh
│  │  ├─ dnadiffusion_run.sh
│  │  ├─ install_mambaforge.sh
│  │  ├─ README.md
│  │  ├─ slurm_interactive.sh
│  │  └─ test_path.sh
│  └─ conda
│     └─ environment.yml
├─ LICENSE.md
├─ Makefile
├─ mkdocs.yml
├─ notebooks
│  ├─ experiments
│  │  ├─ conditional_diffusion
│  │  │  ├─ accelerate_diffusion_conditional_4_cells.ipynb
│  │  │  ├─ dna_diff_baseline_conditional_UNET.ipynb
│  │  │  ├─ dna_diff_baseline_conditional_UNET_with_time_warping.ipynb
│  │  │  ├─ easy_training_Conditional_Code_to_refactor_UNET_ANNOTATED_v4 (2).ipynb
│  │  │  ├─ full_script_version_from_accelerate_notebook
│  │  │  │  ├─ dnadiffusion.py
│  │  │  │  ├─ filter_data.ipynb
│  │  │  │  ├─ master_dataset.ipynb
│  │  │  │  └─ README.MD
│  │  │  ├─ previous_version
│  │  │  │  └─ Conditional_Code_to_refactor_UNET_ANNOTATED_v3 (2).ipynb
│  │  │  ├─ README.MD
│  │  │  ├─ vq_vae_accelerate_diffusion_conditional_4_cells.ipynb
│  │  │  └─ VQ_VAE_LATENT_SPACE_WITH_METRICS.ipynb
│  │  └─ README.md
│  ├─ README.md
│  ├─ refactoring
│  │  └─ README.md
│  └─ tutorials
│     └─ README.md
├─ pyproject.toml
├─ README.md
├─ src
│  ├─ dnadiffusion
│  │  ├─ callbacks
│  │  │  ├─ ema.py
│  │  │  └─ sampling.py
│  │  ├─ cli
│  │  │  └─ __init__.py
│  │  ├─ configs.py
│  │  ├─ data
│  │  │  ├─ dataloader.py
│  │  │  ├─ encode_data.npy
│  │  │  ├─ encode_data.pkl
│  │  │  ├─ K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt
│  │  │  ├─ model4cells_train_split_3_50_dims.pkl
│  │  │  ├─ README.md
│  │  │  └─ __init__.py
│  │  ├─ losses
│  │  │  ├─ README.md
│  │  │  └─ __init__.py
│  │  ├─ metrics
│  │  │  ├─ README.md
│  │  │  ├─ sampling_metrics.py
│  │  │  ├─ validation_metrics.py
│  │  │  └─ __init__.py
│  │  ├─ models
│  │  │  ├─ diffusion.py
│  │  │  ├─ modules.py
│  │  │  ├─ networks.py
│  │  │  ├─ README.md
│  │  │  ├─ training_modules.py
│  │  │  ├─ unet.py
│  │  │  └─ __init__.py
│  │  ├─ README.md
│  │  ├─ sample.py
│  │  ├─ trainer.py
│  │  ├─ utils
│  │  │  ├─ ema.py
│  │  │  ├─ README.md
│  │  │  ├─ scheduler.py
│  │  │  ├─ utils.py
│  │  │  └─ __init__.py
│  │  ├─ __about__.py
│  │  ├─ __init__.py
│  │  └─ __main__.py
│  └─ refactor
│     ├─ config.py
│     ├─ configs
│     │  ├─ callbacks
│     │  │  └─ default.yaml
│     │  ├─ data
│     │  │  ├─ sequence.yaml
│     │  │  └─ vanilla_sequences.yaml
│     │  ├─ logger
│     │  │  └─ wandb.yaml
│     │  ├─ main.yaml
│     │  ├─ model
│     │  │  ├─ dnaddpmdiffusion.yaml
│     │  │  ├─ dnadiffusion.yaml
│     │  │  ├─ lr_scheduler
│     │  │  │  └─ MultiStepLR.yaml
│     │  │  ├─ optimizer
│     │  │  │  └─ adam.yaml
│     │  │  └─ unet
│     │  │     ├─ unet.yaml
│     │  │     └─ unet_conditional.yaml
│     │  ├─ paths
│     │  │  └─ default.yaml
│     │  └─ trainer
│     │     ├─ ddp.yaml
│     │     └─ default.yaml
│     ├─ data
│     │  ├─ sequence_dataloader.py
│     │  └─ sequence_datamodule.py
│     ├─ main.py
│     ├─ models
│     │  ├─ diffusion
│     │  │  ├─ ddpm.py
│     │  │  └─ diffusion.py
│     │  ├─ encoders
│     │  │  └─ vqvae.py
│     │  └─ networks
│     │     ├─ unet_lucas.py
│     │     └─ unet_lucas_cond.py
│     ├─ README.md
│     ├─ sample.py
│     ├─ tests
│     │  ├─ data
│     │  │  └─ test_sequence_dataloader.py
│     │  ├─ models
│     │  │  ├─ diffusion
│     │  │  │  ├─ test_ddim.py
│     │  │  │  └─ test_ddpm.py
│     │  │  ├─ encoders
│     │  │  │  └─ test_vqvae.py
│     │  │  └─ networks
│     │  │     ├─ test_unet_bitdiffusion.py
│     │  │     └─ test_unet_lucas.py
│     │  ├─ utils
│     │  │  ├─ test_ema.py
│     │  │  ├─ test_misc.py
│     │  │  ├─ test_network.py
│     │  │  └─ test_schedules.py
│     │  └─ __init__.py
│     └─ utils
│        ├─ ema.py
│        ├─ metrics.py
│        ├─ misc.py
│        ├─ network.py
│        └─ schedules.py
├─ tests
│  ├─ conftest.py
│  ├─ test_add.py
│  ├─ test_main.py
│  └─ __init__.py
├─ test_environment.py
└─ train.py

```

## Testing

dnadiffusion will be tested using the [pytest](https://docs.pytest.org/en/stable/) framework.

Endpoint to execute all the tests can be found in the Makefile.

## Deployment and Maintenance

dnadiffusion will be distributed as a python package that can be installed and executed on any system with python version 3.10 or greater.

Endpoints to create the package and distribute it can be found in the Makefile.