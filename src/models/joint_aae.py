import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from .layers import Encoder, Decoder, Discriminator 
from .utils_deep import Optimisation_AAE
import numpy as np
from torch.autograd import Variable 
class AAE(nn.Module, Optimisation_AAE):
    
    def __init__(self, input_dims, config):

        '''
        
        Initialise Adversarial Autoencoder model.

        input_dims: The input data dimension.
        config: Configuration dictionary.

        
        '''

        super().__init__()
        self._config = config
        self.model_type = 'joint_AAE'
        self.input_dims = input_dims
        self.hidden_layer_dims = config['hidden_layers'].copy()
        self.z_dim = config['latent_size']
        self.hidden_layer_dims.append(self.z_dim)
        self.non_linear = config['non_linear']
        self.SNP_model = config['SNP_model']
        self.learning_rate = config['learning_rate']
        self.n_views = len(input_dims)

        self.joint_representation = True

        self.encoders = torch.nn.ModuleList([Encoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=False, non_linear=self.non_linear) for input_dim in self.input_dims])
        self.decoders = torch.nn.ModuleList([Decoder(input_dim = input_dim, hidden_layer_dims=self.hidden_layer_dims, variational=False, non_linear=self.non_linear) for input_dim in self.input_dims])
        self.discriminator = Discriminator(input_dim = self.z_dim, hidden_layer_dims=[3], output_dim=1)
        
        self.encoder_optimizers = [torch.optim.Adam(list(self.encoders[i].parameters()), lr=self.learning_rate) for i in range(self.n_views)]
        self.generator_optimizers = [torch.optim.Adam(list(self.encoders[i].parameters()), lr=self.learning_rate) for i in range(self.n_views)]
        self.decoder_optimizers = [torch.optim.Adam(list(self.decoders[i].parameters()), lr=self.learning_rate) for i in range(self.n_views)]
        self.discriminator_optimizer = torch.optim.Adam(list(self.discriminator.parameters()), lr=self.learning_rate)
    
    def encode(self, x):
        z = []
        for i in range(self.n_views):
            z_ = self.encoders[i](x[i])
            z.append(z_)
            
        z = torch.stack(z)
        mean_z = torch.mean(z, axis=0)
        return z
    

    def decode(self, z):
        x_out = []
        for i in range(self.n_views):
            for j in range(self.n_views):
                x_ = self.decoders[i](z)
                x_out.append(x_)
        return x_out

    def disc(self, z):
        z_real = Variable(torch.randn(z[0].size()[0], self.z_dim) * 1.).to(self.device)
        d_real = self.discriminator(z_real)
        d_fake = self.discriminator(z)
        return d_real, d_fake

    def forward_recon(self, x):
        z = self.encode(x)
        x_out = self.decode(z)
        fwd_rtn = {'x_out': x_out,
                    'z': z}
        return fwd_rtn
    
    def forward_discrim(self, x):
        [encoder.eval() for encoder in self.encoders]
        z = self.encode(x)
        d_real, d_fake = self.disc(z)
        fwd_rtn = {'d_real': d_real,
                    'd_fake': d_fake,
                    'z': z}
        return fwd_rtn

    
    def forward_gen(self, x):
        [encoder.train() for encoder in self.encoders]
        self.discriminator.eval()
        z = self.encode(x)
        _, d_fake = self.disc(z)
        fwd_rtn = {'d_fake': d_fake,
                    'z': z}
        return fwd_rtn

    @staticmethod
    def recon_loss(self, x, fwd_rtn):
        x_out = fwd_rtn['x_out']
        recon_loss = 0
        for i in range(self.n_views):
            recon_loss+= torch.mean(((x_out[i] - x[i])**2).sum(dim=-1))
        return recon_loss/self.n_views

    @staticmethod
    def generator_loss(self, fwd_rtn):
        z = fwd_rtn['z']
        d_fake = fwd_rtn['d_fake']
        gen_loss= -torch.mean(torch.log(d_fake+self.eps))
        return gen_loss

    @staticmethod
    def discriminator_loss(self, fwd_rtn):
        z = fwd_rtn['z']
        d_real = fwd_rtn['d_real']
        d_fake = fwd_rtn['d_fake']

        disc_loss= -torch.mean(torch.log(d_real+self.eps)+torch.log(1-d_fake+self.eps))

        return disc_loss
