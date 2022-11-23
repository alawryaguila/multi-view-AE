import torch
import torch.nn.functional as F
from torchvision import datasets, transforms
from collections import OrderedDict
from multiviewae.models import mmVAE
import matplotlib.pyplot as plt
import umap
import pandas as pd

#Load the MNIST data
MNIST_1 = datasets.MNIST('./data/MNIST', train=True, download=True, transform=transforms.Compose([
        transforms.ToTensor()]))

data_img = MNIST_1.train_data.reshape(-1,784).float()/255.
target = MNIST_1.targets
data_txt = F.one_hot(target, 10)
data_txt = data_txt.type(torch.FloatTensor)

MNIST_1 = datasets.MNIST('./data/MNIST', train=False, download=True, transform=transforms.Compose([
        transforms.ToTensor()
    ]))

data_img_test = MNIST_1.train_data.reshape(-1,784).float()/255.
target = MNIST_1.targets
data_txt_test = F.one_hot(target, 10)
data_txt_test = data_txt_test.type(torch.FloatTensor)

#Define parameters
input_dims=[784, 10]
max_epochs = 50
batch_size = 2000
latent_dim = 20

#Define model
mmvae = mmVAE(
        cfg="./config/MNIST_image_text.yaml",
        input_dim=input_dims,
        z_dim=latent_dim,
    )

#Train the model
print('fit mmvae')
mmvae.fit(data_img, data_txt,  max_epochs=max_epochs, batch_size=batch_size)

#Predict the latents and reconstructions
latent = mmvae.predict_latents(data_img_test, data_txt_test)
pred = mmvae.predict_reconstruction(data_img_test, data_txt_test)

data_sample = data_img[20]

#indices: view 1 latent, view 1 decoder, sample 21
pred_sample = pred[0][0][20]


fig, axarr = plt.subplots(1, 2)
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
axarr[0].imshow(data_sample.reshape(28,28))
axarr[1].imshow(pred_sample.reshape(28,28))

colors = {0: 'tab:blue', 1:'tab:orange', 2: 'r', 3: 'c', 4: 'm', 5: 'y',
6: 'g', 7: 'k', 8: 'tab:pink', 9: 'tab:gray'}

latent_0, latent_1 = latent[0], latent[1]

reducer = umap.UMAP(random_state=42)
projections = reducer.fit_transform(latent_0)

fig=plt.figure(figsize=(8,6)) 
ax1 = fig.add_subplot(1, 1, 1)
ax1.scatter(projections[:,0], projections[:,1], c=pd.Series(target).map(colors))
ax1.set_title('mmVAE UMAP latent 0')
handles, labels = plt.gca().get_legend_handles_labels()
by_label = OrderedDict(zip(labels, handles))
plt.legend(by_label.values(), by_label.keys())
plt.tight_layout()
plt.show()