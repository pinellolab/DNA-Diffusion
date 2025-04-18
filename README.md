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

---

**Documentation**: <a href="https://pinellolab.github.io/DNA-Diffusion" target="_blank">https://pinellolab.github.io/DNA-Diffusion</a>

**Source Code**: <a href="https://github.com/pinellolab/DNA-Diffusion" target="_blank">https://github.com/pinellolab/DNA-Diffusion</a>

---

## Introduction

DNA-Diffusion is diffusion-based model for generation of 200bp cell type-specific synthetic regulatory elements.

<div align="center">
<img src="docs/images/dnadiffusion.png" width="600"/>
</div>




## Installation
Our preferred package / project manager is [uv](https://github.com/astral-sh/uv).
To install the necessary packages, run:

```bash
uv sync
```
This will create a virtual environment in `.venv` and install all dependencies listed in the pyproject.toml file.

## Usage

### Sequence Generation
We provide a basic config file for generating sequences using the diffusion model resulting in 1000 sequences made per cell type. Base generation utilizes a guidance scale 1.0, however this can be tuned within the sample.py with the `cond_weight_to_metric` parameter. To generate sequences call:

```bash
uv run sample.py
```


### Training
If you would like to train the model, we provide a basic config file for training the diffusion model. To train the model call:

```bash
uv run train.py
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
