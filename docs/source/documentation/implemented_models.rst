Models implemented
==================

Below is a table with the models contained within this repository and links to the original papers.
     
+------------+---------------------------------------------------------------------------------------------+------------------+
| Model class| Model name                                                                                  | Number of views  |
+============+==============================================================================================+==================+
| mcVAE       | Multi-Channel Variational Autoencoder (mcVAE) [1]                                           | >=1              |
+------------+----------------------------------------------------------------------------------------------+------------------+
| AE          | Multi-view Autoencoder                                                                      |   >=1            |
+------------+----------------------------------------------------------------------------------------------+------------------+
| AAE         | Multi-view Adversarial Autoencoder with separate latent representations                     |    >=1           |
+------------+----------------------------------------------------------------------------------------------+------------------+
| DVCCA       | Deep Variational CCA [3]                                                                    |    2             |
+------------+----------------------------------------------------------------------------------------------+------------------+
| jointAAE    | Multi-view Adversarial Autoencoder with joint latent representation                         |   >=1            |
+------------+----------------------------------------------------------------------------------------------+------------------+
| wAAE        | Multi-view Adversarial Autoencoder with joint latent representation and wasserstein loss    |    >=1           |
+------------+----------------------------------------------------------------------------------------------+------------------+
| mmVAE       | Variational mixture-of-experts autoencoder (MMVAE) [4]                                      |   >=1            |
+------------+----------------------------------------------------------------------------------------------+------------------+
| mVAE        | Multimodal Variational Autoencoder (MVAE) [5]                                               |    >=1           |
+------------+----------------------------------------------------------------------------------------------+------------------+
| me_mVAE     | Multimodal Variational Autoencoder (MVAE) with separate ELBO terms for each view [5]        |    >=1           |
+------------+----------------------------------------------------------------------------------------------+------------------+
| JMVAE       |  Joint Multimodal Variational Autoencoder(JMVAE-kl) [6]                                     |    2             |
+------------+----------------------------------------------------------------------------------------------+------------------+
| MVTCAE      | Multi-View Total Correlation Auto-Encoder (MVTCAE) [8]                                      |    >=1           |
+------------+----------------------------------------------------------------------------------------------+------------------+
| MoPoEVAE    |  Mixture-of-Products-of-Experts VAE [7]                                                     |    >=1           |
+------------+----------------------------------------------------------------------------------------------+------------------+
| mmJSD       |  Multimodal Jensen-Shannon divergence model (mmJSD) [9]                                     |    >=1           |
+------------+----------------------------------------------------------------------------------------------+------------------+
|weighted_mVAE|  Generalised Product-of-Experts Variational Autoencoder (gPoE-MVAE) [10,11]                 |    >=1           |
+------------+----------------------------------------------------------------------------------------------+------------------+
| VAE_barlow  | Multi-view Variational Autoencoder with barlow twins loss between latents. [12,13]          |    2             |
+------------+----------------------------------------------------------------------------------------------+------------------+
| AE_barlow   | Multi-view Autoencoder with barlow twins loss between latents. [12,13]                      |    2             |
+------------+----------------------------------------------------------------------------------------------+------------------+

[1] Antelmi, Luigi & Ayache, Nicholas & Robert, Philippe & Lorenzi, Marco. (2019). Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data. 

[2] Sohn, K., Lee, H., & Yan, X. (2015). Learning Structured Output Representation using Deep Conditional Generative Models. NIPS.

[3] Wang, Weiran & Lee, Honglak & Livescu, Karen. (2016). Deep Variational Canonical Correlation Analysis.

[4] Yuge Shi, N. Siddharth, Brooks Paige, and Philip H. S. Torr. 2019. Variational mixture-of-experts autoencoders for multi-modal deep generative models. Proceedings of the 33rd International Conference on Neural Information Processing Systems. Curran Associates Inc., Red Hook, NY, USA, Article 1408, 15718–15729.

[5] Wu, Mike & Goodman, Noah. (2018). Multimodal Generative Models for Scalable Weakly-Supervised Learning. 

[6] Suzuki, Masahiro and Nakayama, Kotaro and Matsuo, Yutaka. (2016). Joint Multimodal Learning with Deep Generative Models.

[7] Sutter, Thomas & Daunhawer, Imant & Vogt, Julia. (2021). Generalized Multimodal ELBO. 

[8] Hwang, HyeongJoo and Kim, Geon-Hyeong and Hong, Seunghoon and Kim, Kee-Eung. Multi-View Representation Learning via Total Correlation Objective. 2021. NeurIPS

[9] Sutter, Thomas & Daunhawer, Imant & Vogt, Julia. (2021). Multimodal Generative Learning Utilizing Jensen-Shannon-Divergence. Advances in Neural Information Processing Systems. 33. 

[10] Lawry Aguila, A., Chapman, J., Altmann, A. (2023). Multi-modal Variational Autoencoders for normative modelling across multiple imaging modalities. arXiv

[11] Cao, Y., & Fleet, D. (2014). Generalized Product of Experts for Automatic and Principled Fusion of Gaussian Process Predictions. arXiv.

[12] Zbontar, J., Jing, L., Misra, I., LeCun, Y., & Deny, S. (2021). Barlow Twins: Self-Supervised Learning via Redundancy Reduction. International Conference on Machine Learning.

[13] Chapman et al., (2021). CCA-Zoo: A collection of Regularized, Deep Learning based, Kernel, and Probabilistic CCA methods in a scikit-learn style framework. Journal of Open Source Software, 6(68), 3823, https://doi.org/10.21105/joss.03823
