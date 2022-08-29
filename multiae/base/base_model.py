import os
import ast
import numpy as np
import hydra

import torch
import pytorch_lightning as pl

from os.path import join, exists, isdir
from abc import ABC, abstractmethod
from hydra import compose, initialize, initialize_config_dir

from torch.utils.data.dataloader import DataLoader
from omegaconf import OmegaConf

from .constants import MODELS, VARIATIONAL_MODELS, SPARSE_MODELS, CONFIG_KEYS
from .dataloaders import MultiviewDataModule
from .datasets import MVDataset
from datetime import datetime

class BaseModelAE(ABC, pl.LightningModule):
    is_variational = False

    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        assert(model_name is not None)  # have to choose which model always
        assert(input_dim is not None)



        super().__init__()
        self.model_name = model_name

        with initialize(version_base=None, config_path="../configs"):
            def_cfg = compose(
                            config_name="default",
                            return_hydra_config=True,
                            overrides=[f"model_type={self.model_name}.yaml"]
                        )

        if cfg is not None: # user overrides default config
            workdir = os.getcwd()   # TODO: assumes path is relative to working dir
            with initialize_config_dir(version_base=None, config_dir=workdir):
                user_cfg = compose(
                            config_name=cfg,
                            return_hydra_config=True
                        )
            def_cfg = self.__updateconfig(def_cfg, user_cfg)

        # some variables should not be set for certain models
        self.cfg = self.__checkconfig(def_cfg)
        print("MODEL: ", self.model_name)
        self.print_config()

        self.__dict__.update(self.cfg.model)

        if all(k in self.cfg.model for k in ["seed_everything", "seed"]):
            pl.seed_everything(self.cfg.model.seed, workers=True)
            

        assert isinstance(input_dim,list), 'input_dim must be a list of input dimensions'
        assert (isinstance(dim, int) for dim in input_dim), 'Input dimensions must be integers'
        assert isinstance(z_dim, int), 'z_dim must be an integer'
        
        self.input_dim = input_dim 
        if z_dim is not None:   # overrides hydra config... passed arg has precedence
            self.z_dim = z_dim  
            self.cfg.model.z_dim = z_dim
        self.n_views = len(self.input_dim)

        self._setencoders()
        self._setdecoders()

        # TODO: should this be in the end of instance init()?
        self.save_hyperparameters()
        self.create_folder(self.cfg.out_dir)
        run_time = datetime.now().strftime("%Y-%m-%d_%H%M")
        OmegaConf.save(self.cfg, join(self.cfg.out_dir, 'config_{0}.yaml'.format(run_time))) #TODO only save model parameters
        
    ################################            public methods
    def fit(self, *data, labels=None, max_epochs=None, batch_size=None):
        self._training = True
        data = list(data) 
        if max_epochs is not None:
            self.max_epochs = max_epochs
            self.cfg.trainer.max_epochs = max_epochs
        else:
            self.max_epochs = self.cfg.trainer.max_epochs

        if batch_size is not None:
            self.batch_size = batch_size
            self.cfg.datamodule.batch_size = batch_size
        else:
            self.batch_size = self.cfg.datamodule.batch_size

        # check len(data) == n_views and data_dim == input_dim
        assert(len(data) == self.n_views)
        for i in range(self.n_views):
            assert(data[i].shape[1] == self.input_dim[i])   # TODO: this is only for 1D input

        callbacks = []
        if self.cfg.datamodule.is_validate:
            for _, cb_conf in self.cfg.callbacks.items():
                callbacks.append(hydra.utils.instantiate(cb_conf))

        logger = hydra.utils.instantiate(self.cfg.logger)

        # NOTE: have to check file exists otherwise error raised
        if (self.cfg.trainer.resume_from_checkpoint is None) or \
            (not os.path.exists(self.cfg.trainer.resume_from_checkpoint)):
            self.cfg.trainer.resume_from_checkpoint = None
        py_trainer = hydra.utils.instantiate(
            self.cfg.trainer, callbacks=callbacks, logger=logger
        )

        datamodule = hydra.utils.instantiate(
           self.cfg.datamodule, data=data, labels=labels, _convert_="all"
        )

        py_trainer.fit(self, datamodule)

    def predict_latents(self, *data, batch_size=None):
        return self.__predict(*data, batch_size=batch_size)

    def predict_reconstruction(self, *data, batch_size=None):
        return self.__predict(*data, batch_size=batch_size, is_recon=True)

    def print_config(self, keys=None):
        if keys is not None:
            for k in keys:
                if k in CONFIG_KEYS:
                    str = (OmegaConf.to_yaml(self.cfg[k])).replace("\n", "\n  ")
                    print(f"{k}:\n  {str}")
        else:
            self.print_config(keys=CONFIG_KEYS)
    
    def create_folder(self, dir_path):
        check_folder = isdir(dir_path)
        if not check_folder:
            os.makedirs(dir_path)

    ################################            abstract methods
    # TODO: should probably have defaults?
    @abstractmethod
    def encode(self, x):
        raise NotImplementedError()

    @abstractmethod
    def decode(self, z):
        raise NotImplementedError()

    @abstractmethod
    def loss_function(self, x, fwd_rtn):
        raise NotImplementedError()

    ################################            LightningModule methods
    @abstractmethod
    def forward(self, x):
         raise NotImplementedError()

    def training_step(self, batch, batch_idx, optimizer_idx=None):
        return self.__step(batch, batch_idx, stage="train")

    def validation_step(self, batch, batch_idx):
        return self.__step(batch, batch_idx, stage="val")

    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.cfg.out_dir, "model.ckpt"))
        torch.save(self, join(self.cfg.out_dir, "model.pkl"))

    def configure_optimizers(self):
        optimizers = [
            torch.optim.Adam(
                list(self.encoders[i].parameters())
                + list(self.decoders[i].parameters()),
                lr=self.learning_rate,
            )
            for i in range(self.n_views)
        ]
        return optimizers

    ################################            protected methods, can be overwritten by child
    def _setencoders(self):
        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.encoder,
                    input_dim=d,    # TODO: check if multidim. MLP does not support multidim.
                    z_dim=self.z_dim,
                    _recursive_=False,
                    _convert_ = "all"
                )
                for d in self.input_dim
            ]
        )

    def _setdecoders(self):
        self.decoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.decoder,
                    input_dim=d,
                    z_dim=self.z_dim,
                    _recursive_=False,
                    _convert_ = "all"
                )
                for d in self.input_dim
            ]
        )

    ################################            private methods
    def __updateconfig(self, orig, update):
        # TODO: except _target_
        update_keys = list(set(update.keys()) & set(CONFIG_KEYS))
        for k in update_keys:
            for key, val in update[k].items():
                if key in orig[k].keys():
                    orig[k][key] = val
        return orig

    def __checkconfig(self, cfg):
        
        assert self.model_name in MODELS, "Model name is invalid"
        
        if self.model_name in VARIATIONAL_MODELS:
            self.is_variational = True
            #TODO encoder must be variational

        # should be always false for non-sparse models
        if self.model_name not in SPARSE_MODELS:
            cfg.model.sparse = False
        # else configurable
        #TODO if sparse prior must be normal dist
        return cfg

    # TODO: batch_idx is not manual_seed --> what does this mean?
    def __step(self, batch, batch_idx, stage):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)
        for loss_n, loss_val in loss.items():
            self.log(
                f"{stage}_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def __predict(self, *data, batch_size=None, is_recon=False):
        self._training = False

        data = list(data)

        # check len(data) == n_views and data_dim == input_dim
        assert(len(data) == self.n_views)
        for i in range(self.n_views):
            assert(data[i].shape[1] == self.input_dim[i])   # TODO: this is only for 1D input

        dataset = MVDataset(data, labels=None)

        if batch_size is None:
            batch_size = data[0].shape[0]

        generator = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        with torch.no_grad():
            z_ = None
            for batch_idx, local_batch in enumerate(generator):
                local_batch = [
                    local_batch_.to(self.device) for local_batch_ in local_batch
                ]
                z = self.encode(local_batch)
                if self.sparse:
                    z = self.apply_threshold(z)
                if is_recon:
                    z = self.decode(z)

                z = [
                        [ d__._sample().cpu().detach().numpy() for d__ in d_ ]
                        if isinstance(d_, (list))
                        else
                        (d_.cpu().detach().numpy() if isinstance(d_, torch.Tensor)
                        else d_._sample().cpu().detach().numpy())
                        for d_ in z
                    ]

                if z_ is not None:
                    z_ = [
                            [ np.append(d_, d, axis=0) for d_, d in zip(p_,p) ]
                            if isinstance(p_, list) else np.append(p_, p, axis=0)
                            for p_, p in zip(z_, z)
                         ]
                else:
                    z_ = z
        return z_


################################################################################
class BaseModelVAE(BaseModelAE):

    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=model_name,
                cfg=cfg,
                input_dim=input_dim,
                z_dim=z_dim)

    ################################            class methods
    def apply_threshold(self, z):
        """
        Implementation from: https://github.com/ggbioing/mcvae
        """

        assert self.threshold <= 1.0
        keep = (self.__dropout() < self.threshold).squeeze().cpu()
        z_keep = []

        for _ in z:
            _ = _._sample()
            _[:, ~keep] = 0
            d = hydra.utils.instantiate(
                self.cfg.encoder.enc_dist, loc=_, scale=1
            )
            z_keep.append(d)
            del _

        return z_keep

    ################################            protected methods
    def _setencoders(self):

        if self.sparse and self.threshold != 0.:
            self.log_alpha = torch.nn.Parameter(
                torch.FloatTensor(1, self.z_dim).normal_(0, 0.01)
            )
        else:
            self.sparse = False
            self.log_alpha = None

        self.encoders = torch.nn.ModuleList(
            [
                hydra.utils.instantiate(
                    self.cfg.encoder,
                    input_dim=d,
                    z_dim=self.z_dim,
                    sparse=self.sparse,
                    log_alpha=self.log_alpha,
                    _recursive_=False,
                    _convert_="all"
                )
                for d in self.input_dim
            ]
        )

    ################################            private methods
    def __dropout(self):
        """
        Implementation from: https://github.com/ggbioing/mcvae
        """
        alpha = torch.exp(self.log_alpha.detach())
        return alpha / (alpha + 1)

################################################################################
class BaseModelAAE(BaseModelAE):
    is_wasserstein = False

    @abstractmethod
    def __init__(
        self,
        model_name = None,
        cfg = None,
        input_dim = None,
        z_dim = None
    ):

        super().__init__(model_name=model_name,
                        cfg=cfg,
                        input_dim=input_dim,
                        z_dim=z_dim)

        self.automatic_optimization = False

        # TODO - check discriminator dimensionality is correct (nviews + 1?) and for joint model too
        self.discriminator = hydra.utils.instantiate(
            self.cfg.discriminator,
            input_dim=self.z_dim,
            output_dim=(self.n_views + 1),
            is_wasserstein=self.is_wasserstein,
            _convert_="all"
        )

    ################################            abstract methods
    @abstractmethod
    def forward_recon(self, x):
        raise NotImplementedError()

    @abstractmethod
    def forward_discrim(self, x):
        raise NotImplementedError()

    @abstractmethod
    def forward_gen(self, x):
        raise NotImplementedError()

    @abstractmethod
    def recon_loss(self, x, fwd_rtn):
        raise NotImplementedError()

    @abstractmethod
    def generator_loss(self, x, fwd_rtn):
        raise NotImplementedError()

    @abstractmethod
    def discriminator_loss(self, x, fwd_rtn):
        raise NotImplementedError()

    ################################        unused abstract methods
    def loss_function(self, x, fwd_rtn):
        pass

    def forward(self, x):
        pass

    ################################            LightningModule methods
    def training_step(self, batch, batch_idx):
        loss = self.__optimise_batch(batch)
        for loss_n, loss_val in loss.items():
            self.log(
                f"train_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def validation_step(self, batch, batch_idx):
        loss = self.__validate_batch(batch)
        for loss_n, loss_val in loss.items():
            self.log(
                f"val_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            )
        return loss["loss"]

    def configure_optimizers(self):
        optimizers = []
        [   # TODO: why the brackets?
            optimizers.append(
                torch.optim.Adam(
                    list(self.encoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.decoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        [
            optimizers.append(
                torch.optim.Adam(
                    list(self.encoders[i].parameters()), lr=self.learning_rate
                )
            )
            for i in range(self.n_views)
        ]
        optimizers.append(
            torch.optim.Adam(
                list(self.discriminator.parameters()), lr=self.learning_rate
            )
        )
        return optimizers

    ################################            private methods
    def __optimise_batch(self, local_batch):
        fwd_return = self.forward_recon(local_batch)
        loss_recon = self.recon_loss(local_batch, fwd_return)
        opts = self.optimizers()
        enc_opt = [opts.pop(0) for idx in range(self.n_views)]
        dec_opt = [opts.pop(0) for idx in range(self.n_views)]
        gen_opt = [opts.pop(0) for idx in range(self.n_views)]
        disc_opt = opts[0]
        [optimizer.zero_grad() for optimizer in enc_opt]
        [optimizer.zero_grad() for optimizer in dec_opt]
        self.manual_backward(loss_recon)
        [optimizer.step() for optimizer in enc_opt]
        [optimizer.step() for optimizer in dec_opt]

        fwd_return = self.forward_discrim(local_batch)
        loss_disc = self.discriminator_loss(fwd_return)
        disc_opt.zero_grad()
        self.manual_backward(loss_disc)
        disc_opt.step()
        if self.is_wasserstein:
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        fwd_return = self.forward_gen(local_batch)
        loss_gen = self.generator_loss(fwd_return)
        [optimizer.zero_grad() for optimizer in gen_opt]
        self.manual_backward(loss_gen)
        [optimizer.step() for optimizer in gen_opt]
        loss_total = loss_recon + loss_disc + loss_gen
        loss = {
            "loss": loss_total,
            "recon": loss_recon,
            "disc": loss_disc,
            "gen": loss_gen,
        }
        return loss

    def __validate_batch(self, local_batch):
        with torch.no_grad():
            self.eval()
            fwd_return = self.forward_recon(local_batch)
            loss_recon = self.recon_loss(local_batch, fwd_return)
            fwd_return = self.forward_discrim(local_batch)
            loss_disc = self.discriminator_loss(fwd_return)
            fwd_return = self.forward_gen(local_batch)
            loss_gen = self.generator_loss(fwd_return)
            loss_total = loss_recon + loss_disc + loss_gen
            loss = {
                "loss": loss_total,
                "recon": loss_recon,
                "disc": loss_disc,
                "gen": loss_gen,
            }
        return loss

################################################################################
# TODO: use this
class BaseEncoder(pl.LightningModule):

    def __init__(self, **kwargs):
         super().__init__()
         self.save_hyperparameters()

    @abstractmethod
    def forward(self, x):
        pass
        # raise NotImplementedError()

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()

class BaseDecoder(pl.LightningModule):

    def __init__(self, **kwargs):
         super().__init__()
         self.model_name = 'BaseDecoder'
         self.save_hyperparameters()

    @abstractmethod
    def forward(self, z):
        raise NotImplementedError()

    def training_step(self, batch, batch_idx, optimizer_idx):
        raise NotImplementedError()

    def configure_optimizers(self):
        raise NotImplementedError()
