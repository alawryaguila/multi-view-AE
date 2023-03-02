import torch
from torchvision import datasets, transforms
from collections import OrderedDict
from multiviewae import mcVAE, DVCCA
import matplotlib.pyplot as plt #NOTE: matplotlib is not installed with the library and must be installed separately
import pandas as pd

#Load the MNIST data
MNIST_1 = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor(),
    ]))



data_1 = MNIST_1.train_data[:, :, :14].reshape(-1,392).float()/255.
data_2 = MNIST_1.train_data[:, :, 14:].reshape(-1,392).float()/255.
target = MNIST_1.train_labels

MNIST_1 = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))

data_test_1 = MNIST_1.test_data[:, :, :14].reshape(-1,392).float()/255.
data_test_2 = MNIST_1.test_data[:, :, 14:].reshape(-1,392).float()/255.
target_test = MNIST_1.test_labels.numpy()

#Define parameters
input_dim=[392,392]
max_epochs = 500
batch_size = 2000
latent_dim = 2

#Define models
mcvae = mcVAE(
        cfg="./config/example_mnist.yaml",
        input_dim=input_dim,
        z_dim=latent_dim,
    )
dvcca = DVCCA(
        cfg="./config/example_mnist.yaml",
        input_dim=input_dim,
        z_dim=latent_dim,
    )

#Train the models

mcvae.fit(data_1, data_2, max_epochs=max_epochs, batch_size=batch_size)
dvcca.fit(data_1, data_2, max_epochs=max_epochs, batch_size=batch_size)

#Create latent plots - how well does the latent space separate MNIST labels?

mcvae_latent = mcvae.predict_latents(data_test_1, data_test_2)
dvcca_latent = dvcca.predict_latents(data_test_1, data_test_2)

colors = {0: 'tab:blue',
        1:'tab:orange',
        2: 'r',
        3: 'c',
        4: 'm',
        5: 'y',
        6: 'g',
        7: 'k',
        8: 'tab:pink',
        9: 'tab:gray'
        }
fig=plt.figure(figsize=(8,6)) 
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(dvcca_latent[0][:,0], dvcca_latent[0][:,1], c=pd.Series(target_test).map(colors))
ax1.set_title('DVCCA latent vectors')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.tight_layout()
plt.show()

fig=plt.figure(figsize=(8,6)) 
ax1 = fig.add_subplot(1, 2, 1)
ax1.scatter(mcvae_latent[0][:,0], mcvae_latent[0][:,1], c=pd.Series(target_test).map(colors))
ax1.set_title('mcVAE latent vectors view 1')
ax2 = fig.add_subplot(1, 2, 2)
ax2.scatter(mcvae_latent[1][:,0], mcvae_latent[1][:,1], c=pd.Series(target_test).map(colors))
ax2.set_title('mcVAE latent vectors view 2')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.tight_layout()
plt.show()

#mcVAE reconstruction example

pred = mcvae.predict_reconstruction(data_1, data_2)

#Reconstruction plots - how well can the VAE do same view reconstruction?

data_sample = data_1[20]
#indices: view 1 latent, view 1 decoder, sample 21
pred_sample = pred[0][0][20]

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(data_sample.reshape(28,14))
axarr[1].imshow(pred_sample.reshape(28,14))
plt.show()
plt.close()
#Reconstruction plots - how well can the VAE do cross view reconstruction?

data_sample = data_1[20]
#indices: view 1 latent, view 2 decoder, sample 21
pred_sample = pred[0][1][20]

fig, axarr = plt.subplots(1, 2)
axarr[0].imshow(data_sample.reshape(28,14))
axarr[1].imshow(pred_sample.reshape(28,14))
plt.show()
plt.close()

