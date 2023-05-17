![Build Status](https://github.com/alawryaguila/multi-view-ae/actions/workflows/ci.yml/badge.svg)
[![Documentation Status](https://readthedocs.org/projects/multi-view-ae/badge/?version=latest)](https://multi-view-ae.readthedocs.io/en/latest/?badge=latest)
[![Python Version](https://img.shields.io/badge/python-3.7%20%7C%203.8%20%7C%203.9%20%7C%203.10%20-blue)](https://github.com/alawryaguila/multi-view-ae)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05093/status.svg)](https://joss.theoj.org/papers/10.21105/joss.05093)
# Multi-view-AE: Multi-modal subspace learning using autoencoders
<p align="center">
  <img src="https://github.com/alawryaguila/multi-view-AE/blob/master/docs/figures/logo.png" width="600px"></center>
</p>

`multi-view-AE` is a collection of multi-modal autoencoder models for learning joint subspaces from multiple modalities of data. The package is structured such that all models have `fit`, `predict_latent` and `predict_reconstruction` methods. All models are built in Pytorch and Pytorch-Lightning. 

For more information on implemented models and how to use the package, please see the documentation at https://multi-view-ae.readthedocs.io/en/latest/

### Installation
Clone this repository and move to folder:
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

### Contribution guidelines
Contribution guidelines are available at https://multi-view-ae.readthedocs.io/en/latest/
