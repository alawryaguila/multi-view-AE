User Guide
===========

User guide for initialising and running models from the ``multiviewAE`` library. 

Initialise model 
----------

.. code-block:: python

   from multiviewae import mVAE, mcVAE
   from torchvision import datasets, transforms
   
   MNIST_1 = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transforms.Compose([
          transforms.ToTensor(),
   ]))



   data_1 = MNIST_1.train_data[:, :, :14].reshape(-1,392).float()/255.
   data_2 = MNIST_1.train_data[:, :, 14:].reshape(-1,392).float()/255.

   mvae = mVAE(cfg="./examples/config/example_mnist.yaml",
        input_dim=[392, 392],
        z_dim=64)

   mcvae = mcVAE(cfg="./examples/config/example_mnist.yaml",
        input_dim=[392, 392],
        z_dim=64)


The dimensions of the input data, ``input_dim``, must be provided however the path to a configuration file, ``cfg``, and number of latent dimensions, ``z_dim``, are optional. Setting ``z_dim`` will override the value given in the configuration file.

Model fit
----------

.. code-block:: python

   mvae.fit(data_1, data_2,  max_epochs=500, batch_size=1000)
   mcvae.fit(data_1, data_2,  max_epochs=500, batch_size=1000)

When fitting the model, the user must provide input each view of the training data. The user can optionally provide the ``max_epochs`` and ``batch_size``. These would override the settings in the configuration file. 

Model predictions
----------

We can use a trained model to predict the latent dimensions or reconstructions. The structure of the latent and reconstruction list will depend on the type of model. Below shows an example for joint, ``MVAE``,  and coordinate, ``mcVAE``, multi-view VAE models.

.. code-block:: python

   MNIST_1 = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()]))

   data_test_1 = MNIST_1.test_data[:, :, :14].reshape(-1,392).float()/255.
   data_test_2 = MNIST_1.test_data[:, :, 14:].reshape(-1,392).float()/255.

   mvae_latent = mvae.predict_latents(data_test_1, data_test_2)

   mcvae_latent = mcvae.predict_latents(data_test_1, data_test_2)

   mcvae_latent_view1, mcvae_latent_view2 = mcvae_latent[0], mcvae_latent[1]

   mvae_reconstruction = mvae.predict_reconstruction(data_test_1, data_test_2)

   mvae_reconstruction_view1 = mvae_reconstruction[0][0] #view 1 reconstruction from joint latent
   mvae_reconstruction_view2 = mvae_reconstruction[0][1] #view 2 reconstruction from joint latent

   mcvae_reconstruction = mcvae.predict_reconstruction(data_test_1, data_test_2)

   mcvae_reconstruction_view1_latent1 = mcvae_reconstruction[0][0] #view 1 reconstruction from latent 1
   mcvae_reconstruction_view2_latent1 = mcvae_reconstruction[0][1] #view 2 reconstruction from latent 1

   mcvae_reconstruction_view1_latent2 = mcvae_reconstruction[1][0] #view 1 reconstruction from latent 2
   mcvae_reconstruction_view2_latent2 = mcvae_reconstruction[1][1] #view 2 reconstruction from latent 2

Model results
-------------

We can explore the model results, for example using ``matplotlib``. 

.. code-block:: python

   import matplotlib.pyplot as plt #NOTE: matplotlib is not installed with the library and must be installed separately 
   
   #Reconstruction plots - how well can the VAE do same view reconstruction?

   data_sample = data_test_1[20]
   #indices: view 1 latent, view 1 decoder, sample 21
   pred_sample = mcvae_reconstruction_view1_latent1[20]

   fig, axarr = plt.subplots(1, 2)
   axarr[0].imshow(data_sample.reshape(28,14))
   axarr[1].imshow(pred_sample.reshape(28,14))
   plt.show()
   plt.close()
   
   #Reconstruction plots - how well can the VAE do cross view reconstruction?

   #indices: view 1 latent, view 2 decoder, sample 21
   data_sample = data_test_2[20]
   pred_sample = mcvae_reconstruction_view2_latent1[20]

   fig, axarr = plt.subplots(1, 2)
   axarr[0].imshow(data_sample.reshape(28,14))
   axarr[1].imshow(pred_sample.reshape(28,14))
   plt.show()
   plt.close()
   
Model loading   
----------
Trained models can be loaded from the specified path. 

.. code-block:: python

   import torch
   from os.path import join

   #change the path below to your model path
   mvae = torch.load(join('path/to/model', 'model.pkl'))

