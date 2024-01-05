<div align="center">
<img src="https://github.com/alawryaguila/multi-view-AE/blob/master/docs/figures/logo.png" width="500px">

# Multi-modal representation learning using autoencoders

![Build Status](https://github.com/alawryaguila/multi-view-ae/actions/workflows/ci.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/multi-view-ae/badge/?version=latest)](https://multi-view-ae.readthedocs.io/en/latest/?badge=latest)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20-blue)](https://github.com/alawryaguila/multi-view-ae)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05093/status.svg)](https://joss.theoj.org/papers/10.21105/joss.05093)
[![version](https://img.shields.io/pypi/v/multiviewae)](https://pypi.org/project/multiviewae/)
[![codecov](https://codecov.io/gh/alawryaguila/multi-view-AE/graph/badge.svg?token=NKO935MXFG)](https://codecov.io/gh/alawryaguila/multi-view-AE)
[![downloads](https://img.shields.io/pypi/dm/multiviewae)](https://pypi.org/project/multiviewae/)

</div>

`multi-view-AE` is a collection of multi-modal autoencoder models for learning joint representations from multiple modalities of data. The package is structured such that all models have `fit`, `predict_latents` and `predict_reconstruction` methods. All models are built in Pytorch and Pytorch-Lightning. 

Many of the models implemented in the `multi-view-AE` library have been *benchmarked against previous implementations, with equal or improved results*. **See below** for more details.

For more information on implemented models and how to use the package, please see the [documentation](https://multi-view-ae.readthedocs.io/en/latest/).

## Library schematic
<p align="center">
  <img src="https://github.com/alawryaguila/multi-view-AE/blob/master/docs/figures/schematic_diagram.png" width="800px"></center>
</p>

## Models Implemented

Below is a table with the models contained within this repository and links to the original papers.

|Model class   |Model name                                                                                   |Number of views   |Original work|
|:------------:|:-------------------------------------------------------------------------------------------:|:----------------:|:-----------:|
| mcVAE        | Multi-Channel Variational Autoencoder (mcVAE)                                               | >=1              |[link](http://proceedings.mlr.press/v97/antelmi19a.html)|
| AE           | Multi-view Autoencoder                                                                      |   >=1            |               |
| mAAE         | Multi-view Adversarial Autoencoder                                                          |    >=1           |               |
| DVCCA        | Deep Variational CCA                                                                        |    2             |[link](https://arxiv.org/abs/1610.03454)|
| mWAE         | Multi-view Adversarial Autoencoder with a wasserstein loss                                  |    >=1           |               |
| mmVAE        | Variational mixture-of-experts autoencoder (MMVAE)                                          |   >=1            |[link](https://arxiv.org/abs/1911.03393)|
| mVAE         | Multimodal Variational Autoencoder (MVAE)                                                   |    >=1           |[link](https://arxiv.org/abs/1802.05335)|
| me_mVAE      | Multimodal Variational Autoencoder (MVAE) with separate ELBO terms for each view            |    >=1           |[link](https://arxiv.org/abs/1802.05335)|
| JMVAE        |  Joint Multimodal Variational Autoencoder(JMVAE-kl)                                         |    2             |[link](https://arxiv.org/abs/1611.01891)|
| MVTCAE       | Multi-View Total Correlation Auto-Encoder (MVTCAE)                                          |    >=1           |[link](https://proceedings.neurips.cc/paper/2021/file/65a99bb7a3115fdede20da98b08a370f-Paper.pdf)|
| MoPoEVAE     |  Mixture-of-Products-of-Experts VAE                                                         |    >=1           |[link](https://arxiv.org/pdf/2105.02470.pdf)|
| mmJSD        |  Multimodal Jensen-Shannon divergence model (mmJSD)                                         |    >=1           |[link](https://arxiv.org/abs/2006.08242)|
|weighted_mVAE |  Generalised Product-of-Experts Variational Autoencoder (gPoE-MVAE)                         |    >=1           |[link](https://arxiv.org/abs/2303.12706)|
| DMVAE        | Disentangled multi-modal variational autoencoder                                            |    >=1           |[link](https://arxiv.org/abs/2012.13024)|
|weighted_DMVAE| Disentangled multi-modal variational autoencoder with gPoE joint posterior                  |    >=1           |               |
| mmVAEPlus    | Mixture-of-experts multimodal VAE Plus (mmVAE+)                                             |    >=1           |[link](https://openreview.net/forum?id=sdQGxouELX)|

## Installation
To install our package via `pip`:
```bash
pip install multiviewae
```

Or, clone this repository and move to folder:
```bash
git clone https://github.com/alawryaguila/multi-view-AE
cd multi-view-AE
```

Create the customised python environment:
```bash
conda create --name mvae python=3.9
```

Activate python environment:
```bash
conda activate mvae
```

Install the ``multi-view-AE`` package:
```bash
pip install ./
```
## Benchmarking results
To illustrate the efficacy of the ``multi-view-AE`` implementions, we validated some of the implemented models by reproducing a key result of a previous paper. One of the experiments presented in the paper was reproduced using the \texttt{multi-view-AE} implementations using the same network architectures, modelling choices, and training parameters. The code to reproduce the benchmarking experiments is available in the ``benchmarking`` folder. We evaluated performance using the joint log likelihood (↑) and conditional coherence accuracy (↑). Summary of the results of the benchmarking experiments using the BinaryMNIST and PolyMNIST datasets:

|Model   |Experiment |Metric   |Paper| Paper results| ``multi-view-AE`` results| 
|:------------:|:---------:|:----------------:|:-----------:|:-----------:|:-----------:|
| JMVAE        | BinaryMNIST | Joint log likelihood           |[link](https://arxiv.org/abs/1611.01891)|-86.86 | -86.76±0.06 |
| me_mVAE      | BinaryMNIST | Joint log likelihood           |[link](https://arxiv.org/abs/1802.05335)|-86.26 | -86.31±0.08 |
| MoPoEVAE     | PolyMNIST   | Conditional Coherence accuracy |[link](https://proceedings.neurips.cc/paper/2021/file/65a99bb7a3115fdede20da98b08a370f-Paper.pdf)|63/75/79/81 | 68/79/83/84 |
| mmJSD        | PolyMNIST   | Conditional Coherence accuracy |[link](https://proceedings.neurips.cc/paper/2021/file/65a99bb7a3115fdede20da98b08a370f-Paper.pdf)|69/57/64/67 | 75/74/78/80 |
| mmVAE        | PolyMNIST   | Conditional Coherence accuracy |[link](https://proceedings.neurips.cc/paper/2021/file/65a99bb7a3115fdede20da98b08a370f-Paper.pdf)|71/71/71/71 | 71/71/71/71 |
| MVTCAE       | PolyMNIST   | Conditional Coherence accuracy |[link](https://proceedings.neurips.cc/paper/2021/file/65a99bb7a3115fdede20da98b08a370f-Paper.pdf)|59/77/83/86 | 64/81/87/90 |
| mmVAEPlus    | PolyMNIST   | Conditional Coherence accuracy |[link](https://openreview.net/forum?id=sdQGxouELX)| 85.2 | 86.6±0.07 |

## Citation
If you have used `multi-view-AE` in your research, please consider citing our JOSS paper: 

Lawry Aguila et al., (2023). Multi-view-AE: A Python package for multi-view autoencoder models. Journal of Open Source Software, 8(85), 5093, https://doi.org/10.21105/joss.05093

Bibtex entry:
```bibtex
@article{LawryAguila2023, 
doi = {10.21105/joss.05093}, 
url = {https://doi.org/10.21105/joss.05093}, 
year = {2023}, 
publisher = {The Open Journal}, 
volume = {8}, 
number = {85}, 
pages = {5093}, 
author = {Ana Lawry Aguila and Alejandra Jayme and Nina Montaña-Brown and Vincent Heuveline and Andre Altmann}, 
title = {Multi-view-AE: A Python package for multi-view autoencoder models}, journal = {Journal of Open Source Software} 
}
```
## Contribution guidelines
Contribution guidelines are available at https://multi-view-ae.readthedocs.io/en/latest/
