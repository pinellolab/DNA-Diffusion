# DNA Diffusion

<img src='https://raw.githubusercontent.com/pinellolab/DNA-Diffusion/main/docs/images/diff_first_lossy.gif?inline=true'> </img>

<p align="center">
    <em>Generative modeling of regulatory DNA sequences with diffusion probabilistic models.</em>
</p>

[![build](https://github.com/pinellolab/DNA-Diffusion/workflows/Build/badge.svg)](https://github.com/pinellolab/DNA-Diffusion/actions)
[![codecov](https://codecov.io/gh/pinellolab/DNA-Diffusion/branch/main/graph/badge.svg)](https://codecov.io/gh/pinellolab/DNA-Diffusion)
[![PyPI version](https://badge.fury.io/py/dnadiffusion.svg?kill_cache=1)](https://badge.fury.io/py/dnadiffusion)

<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-3-orange.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

[![Model on HF](https://huggingface.co/datasets/huggingface/badges/resolve/main/model-on-hf-sm.svg)](https://huggingface.co/ssenan/DNA-Diffusion)

---

**Documentation**: <a href="https://pinellolab.github.io/DNA-Diffusion" target="_blank">https://pinellolab.github.io/DNA-Diffusion</a>

**Source Code**: <a href="https://github.com/pinellolab/DNA-Diffusion" target="_blank">https://github.com/pinellolab/DNA-Diffusion</a>

---

## Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Recreating data curation, training and sequence generation processes](#recreating-data-curation-training-and-sequence-generation-processes)
  - [Data](#data)
  - [Training](#training)
  - [Model Checkpoint](#model-checkpoint)
  - [Sequence Generation](#sequence-generation)
- [Examples](#examples)
  - [Training Notebook](#training-notebook)
  - [Sequence Generation Notebook](#sequence-generation-notebook)
- [Using your own data](#using-your-own-data)

## Introduction

DNA-Diffusion is diffusion-based model for generation of 200bp cell type-specific synthetic regulatory elements.

<div align="center">
<img src="docs/images/dnadiffusion.png" width="600"/>
</div>


## Installation
Our preferred package / project manager is [uv](https://github.com/astral-sh/uv). Please follow their recommended instructions for installation.

To clone the repository and install the necessary packages, run:

```bash
git clone https://github.com/pinellolab/DNA-Diffusion.git
cd DNA-Diffusion
uv sync
```

This will create a virtual environment in `.venv` and install all dependencies listed in the `uv.lock` file. This is compatible with both CPU and GPU, but preferred operating system is Linux with a recent GPU (e.g. A100 GPU). For detailed versions of the dependencies, please refer to the `uv.lock` file.

## Recreating data curation, training and sequence generation processes

### Data
We provide a small subset of the DHS Index dataset that was used for  training at `data/K562_hESCT0_HepG2_GM12878_12k_sequences_per_group.txt`.

If you would like to recreate the dataset, you can call:

```bash
uv run data/master_dataset_and_filter.py
```
which will download all the necessary data and create a file `data/master_dataset.ftr` containing the full ~3.59 million dataset and a file `data/filtered_dataset.txt` containing the same subset of sequences as above. A rendered version of this code is provided at `notebooks/marimo_master_dataset_and_filter.ipynb`.

### Training
To train the DNA-Diffusion model, we provide a basic config file for training the diffusion model on the same subset of chromatin accessible regions described in the data section above.

To train the model call:

```bash
uv run train.py
```

This runs the model with our predefined config file `configs/train/default.yaml`, which is set to train the model for a minimum of 2000 epochs. The training script will save model checkpoints for the lowest 2 validation loss values in the `checkpoints/` directory. The path to this checkpoint will need to be updated in the sampling config file for sequence generation, as described in the Model Checkpoint section below.

We also provide a base config for debugging that will use a single sequence for training. You can override the default training script to use this debugging config by calling:

```bash
uv run train.py -cn train_debug
```

### Model Checkpoint
We have uploaded the model checkpoint to [HuggingFace](https://huggingface.co/ssenan/DNA-Diffusion).  Below we provide an example script that handles downloading the model checkpoint and loading it for sequence generation.

If you would like to use a model checkpoint generated from the training script above, ensure you update the `checkpoint_path` within the config file `configs/sampling/default.yaml` to point to the location of the model checkpoint. By default, this is set to `checkpoints/model.safetensors`, so you will need to ensure that the model checkpoint is saved in this location. Both `pt` and `safetensors` formats are supported, so you can use either format for the model checkpoint. An example of overriding the checkpoint path from the command line is described in the sequence generation section below.

### Sequence Generation

#### Generate using Hugging Face Checkpoint

We provide a basic config file for generating sequences using the diffusion model resulting in 1000 sequences made per cell type.  To generate sequences using the trained model, you can run the following command:

```bash
uv run sample_hf.py
```

The default setup for sampling will generate 1000 sequences per cell type. You can override the default sampling script to generate one sequence per cell type with the following cli flags:

```bash
uv run sample_hf.py sampling.number_of_samples=1 sampling.sample_batch_size=1
```

Base generation utilizes a guidance scale 1.0, however this can be tuned within the sample.py with the `guidance_scale` parameter. This can be overridden in the command line as follows (generating using guidance scale 7.0):

```bash
uv run sample_hf.py sampling.guidance_scale=7.0 sampling.number_of_samples=1 sampling.sample_batch_size=1
```

Both above examples will generate sequences for all cell types in the dataset. If you would like to generate sequences for a specific cell type, you can do so by specifying the `sampling.cell_type` parameter in the command line. For example, to generate a sequence for the K562 cell type, you can run:

```bash
uv run sample_hf.py data.cell_types=K562 sampling.number_of_samples=1 sampling.sample_batch_size=1
```
or for both K562 and GM12878 cell types, you can run:

```bash
uv run sample_hf.py 'data.cell_types="K562,GM12878"' sampling.number_of_samples=1 sampling.sample_batch_size=1
```
Cell types can be specified as a comma separated string or as a list.

#### Generate using Local Checkpoint

If you would prefer to download the model checkpoint from Hugging Face and use it directly, you can run the following command to download the model and save it in the checkpoint directory:
```bash
wget https://huggingface.co/ssenan/DNA-Diffusion/resolve/main/model.safetensors -O checkpoints/model.safetensors
```

Then you can run the sampling script with the following command:
```
uv run sample.py
```

If you would like to override the checkpoint path from the command line, you can do so with the following command (replacing `checkpoints/model.pt` with the path to your model checkpoint):
```bash
uv run sample.py sampling.checkpoint_path=checkpoints/model.pt
```

## Examples

### Training Notebook

We provide an example colab notebook for training and sampling with the diffusion model. This notebook runs the previous commands for training and sampling.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/160tjHJFtnf-Sj7uyLDBAQjFwUywqOT96?usp=sharing)

along with a copy of the notebook at `notebooks/training_and_sequence_generation.ipynb`

### Sequence Generation Notebook
We also provide a colab notebook for generating sequences with the diffusion model using the trained model hosted on Hugging Face. This notebook runs the previous commands for sampling and shows some example outputs.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1fLjWUkEFka8xyu5Q5uoklnJW7AZ37Tgn?usp=sharing)

along with a copy of the notebook at `notebooks/sequence_generation.ipynb`

Both examples were run on Google Colab using a T4 GPU.

## Using your own data

DNA-Diffusion is designed to be flexible and can be adapted to your own data. To use your own data, you will need to follow these steps:

* Prepare your data in the same format as the DHS Index dataset. The data should be a tab separated text file contains at least the following columns:
  * `chr`: the chromosome of the regulatory element (e.g. chr1, chr2, etc.)
  * `sequence`: the DNA sequence of the regulatory element
  * `TAG`: the cell type of the regulatory element (e.g. K562, hESCT0, HepG2, GM12878, etc.)

additional metadata columns like start, end, continuous accessibility are allowed but not required.

* It's expected that your sequences are 200bp long, however the model can be adapted to work with different sequence lengths by the dataloading code at `src/dnadiffusion/data/dataloader.py`. You can change the `sequence_length` parameter in the function `load_data` to the desired length, but keep in mind that the original model is trained on 200bp sequences so the results may not be as good if you use a different length.
* The model is designed to work with discrete class labels for the cell types, so you will need to ensure that your data is in the same format. If you have continuous labels, you can binarize them into discrete classes using a threshold or some other method. This value is contained within the `TAG` column of the dataset.

The data loading config can be found at `configs/data/default.yaml`, and you can override the default data loading config by passing the `data` parameter to the command line. For example, to use a custom data file, you can run:

```bash
uv run train.py data.data_path=path/to/your/data.txt data.load_saved_data=False
```

It is important to set `data.load_saved_data=False` to ensure that cached data is not used, and instead is regenerated from the provided data file. This will ensure that the model is trained on your own data. This will overwrite the default pkl file, so if you would like to keep the original data, you can set `data.saved_data_path` to a different path. For example:

```bash
uv run train.py data.data_path=path/to/your/data.txt data.load_saved_data=False data.saved_data_path=path/to/your/saved_data.pkl
```


## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tbody>
    <tr>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/LucasSilvaFerreira"><img src="https://avatars.githubusercontent.com/u/5742873?v=4?s=100" width="100px;" alt="Lucas Ferreira da Silva"/><br /><sub><b>Lucas Ferreira da Silva</b></sub></a><br /><a href="#ideas-LucasSilvaFerreira" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pinellolab/DNA-Diffusion/commits?author=LucasSilvaFerreira" title="Code">💻</a></td>
      <td align="center" valign="top" width="14.28%"><a href="http://pinellolab.org"><img src="https://avatars.githubusercontent.com/u/1081322?v=4?s=100" width="100px;" alt="Luca Pinello"/><br /><sub><b>Luca Pinello</b></sub></a><br /><a href="#ideas-lucapinello" title="Ideas, Planning, & Feedback">🤔</a></td>
      <td align="center" valign="top" width="14.28%"><a href="https://github.com/ssenan"><img src="https://avatars.githubusercontent.com/u/8073222?v=4?s=100" width="100px;" alt="Simon"/><br /><sub><b>Simon</b></sub></a><br /><a href="#ideas-ssenan" title="Ideas, Planning, & Feedback">🤔</a> <a href="https://github.com/pinellolab/DNA-Diffusion/commits?author=ssenan" title="Code">💻</a></td>
    </tr>
  </tbody>
</table>

<!-- markdownlint-restore -->
<!-- prettier-ignore-end -->

<!-- ALL-CONTRIBUTORS-LIST:END -->

<a href="https://github.com/pinellolab/DNA-Diffusion/graphs/contributors">
  <img src="https://contributors-img.web.app/image?repo=pinellolab/DNA-Diffusion" />
</a>

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
