# Multi-view autoencoders models 

This repository contains various multi-view autoencoder models built in Pytorch and Pytorch-Lightning.

### Content 
 
 Below is a table with the models contained within this repository and links to the original papers.
 
| Model class      | Model name           | Number of views |
| ------------- |:-------------:| -----:|
| mcVAE      | Multi-Channel Variational Autoencoder (mcVAE) [1] | >=1 |
| AE      | Multi-view Autoencoder    |   >=1 |
| AAE | Multi-view Adversarial Autoencoder with separate latent representations     |    >=1 |
| cVAE     | Conditional Variational Autoencoder [2] | 1 |
| VAE_classifier     | Variational Autoencoder with classifier on latent representation   |   1 |
| DVCCA | Deep Variational CCA [3] |    2 |
|  jointAAE    | Multi-view Adversarial Autoencoder with joint latent representation  |   >=1 |
| wAAE | Multi-view Adversarial Autoencoder with joint latent representation and wasserstein loss    |    >=1 |
|  mmVAE    | Variational mixture-of-experts autoencoder (MMVAE) [4] |   >=1 |
| mVAE | Multi-view VAE with joint representation. Multimodal Variational Autoencoder (MVAE) with ```join_type=PoE``` [5] |    >=1 |
| me_mVAE | Multimodal Variational Autoencoder (MVAE) with separate ELBO terms for each view [5] |    >=1 |
| JMVAE |  Joint Multimodal Variational Autoencoder(JMVAE-kl) [6] |    2 |

[1] Antelmi, Luigi & Ayache, Nicholas & Robert, Philippe & Lorenzi, Marco. (2019). Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data. 

[2] Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models. NIPS.

[3] Wang, Weiran & Lee, Honglak & Livescu, Karen. (2016). Deep Variational Canonical Correlation Analysis.

[4] Yuge Shi, N. Siddharth, Brooks Paige, and Philip H. S. Torr. 2019. Variational mixture-of-experts autoencoders for multi-modal deep generative models. Proceedings of the 33rd International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA, Article 1408, 15718–15729.

[5] Wu, Mike & Goodman, Noah. (2018). Multimodal Generative Models for Scalable Weakly-Supervised Learning. 
 
### Installation

Clone this repository and move to folder:
```bash
git clone https://github.com/alawryaguila/multiAE
cd multiAE
```

Create the customised python environment:
```bash
conda create --name mvm
```

Activate python environment:
```bash
conda activate mvm
```

Install multiAE package:
```bash
python setup.py install
```
