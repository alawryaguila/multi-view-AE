from torch.utils.data.dataloader import DataLoader
from ..utils.dataloaders import MultiviewDataModule
from ..utils.utils import update_dict
import numpy as np
import torch
from ..plot.plotting import Plotting
from os.path import join, exists
import pytorch_lightning as pl
import hydra 
from hydra import compose, initialize
from omegaconf import DictConfig

class BaseModel(pl.LightningModule, Plotting):
    def __init__(self,
        expt=None,
        ):
        super().__init__()
        with initialize(version_base=None, config_path="../configs"):
            if expt:
                self.cfg = compose(config_name="run", return_hydra_config=True, overrides=["experiment={0}.yaml".format(expt)])
            else:
                self.cfg = compose(config_name="run", return_hydra_config=True)

    def fit(self, *data, labels=None, **kwargs):
        self.data = data
        self.labels = labels
        self.val_set = False

        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        self.cfg.callbacks = update_dict(self.cfg.callbacks, kwargs)
        callbacks =  []
        for _, cb_conf in self.cfg.callbacks.items():
            callbacks.append(hydra.utils.instantiate(cb_conf))

        self.cfg.logger = update_dict(self.cfg.logger, kwargs)
        logger = hydra.utils.instantiate(self.cfg.logger)

        self.cfg.trainer = update_dict(self.cfg.trainer, kwargs)
        if not exists(self.cfg.trainer.resume_from_checkpoint):
            self.cfg.trainer.resume_from_checkpoint = None
        py_trainer =  hydra.utils.instantiate(self.cfg.trainer, callbacks=callbacks, logger=logger)

        self.cfg.datamodule = update_dict(self.cfg.datamodule, kwargs)
        datamodule = hydra.utils.instantiate(self.cfg.datamodule, *data, labels=self.labels) 

        py_trainer.fit(self, datamodule)

    def predict_latents(self, *data, val_set=False):
        self.val_set = val_set
        dataset = MultiviewDataModule.dataset(*data, labels=None)
        generator = DataLoader(dataset, batch_size=self.cfg.datamodule.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_idx, local_batch in enumerate(generator):
                local_batch = (
                    [local_batch_.to(self.device) for local_batch_ in local_batch]
                    if isinstance(local_batch, (list, tuple))
                    else local_batch.to(self.device)
                )
                if self.variational:
                    mu, logvar = self.encode(local_batch)
                    pred = self.reparameterise(mu, logvar)
                else:
                    pred = self.encode(local_batch)
                if self.sparse:
                    pred = self.apply_threshold(pred)
                if batch_idx == 0:
                    predictions = self.process_output(pred, data_type="latent")
                else:
                    predictions = self.process_output(
                        pred, pred=predictions, data_type="latent"
                    )
        return predictions

    def process_output(self, data, pred=None, data_type=None):
        if pred is not None:
            if self.cfg.model.variational and data_type is None and self.cfg.model.dist == "gaussian":
                if isinstance(data, (list, tuple)):
                    return [
                        self.process_output(data_, pred=pred_, data_type=data_type)
                        if isinstance(data_, list)
                        else np.append(pred_, self.sample_from_normal(data_), axis=0)
                        for pred_, data_ in zip(pred, data)
                    ]
                return np.append(pred, self.sample_from_normal(data), axis=0)
            if isinstance(data, (list, tuple)):
                return [
                    self.process_output(data_, pred=pred_, data_type=data_type)
                    if isinstance(data_, list)
                    else np.append(pred_, data_, axis=0)
                    for pred_, data_ in zip(pred, data)
                ]
            return np.append(pred, data, axis=0)
        else:
            if self.cfg.model.variational and data_type is None and self.cfg.model.dist == "gaussian":
                if isinstance(data, (list, tuple)):
                    return [
                        self.process_output(data_, data_type=data_type)
                        if isinstance(data_, list)
                        else self.sample_from_normal(data_).cpu().detach().numpy()
                        for data_ in data
                    ]
                return self.sample_from_normal(data).cpu().detach().numpy()
            if isinstance(data, (list, tuple)):
                return [
                    self.process_output(data_, data_type=data_type)
                    if isinstance(data_, list)
                    else data_.cpu().detach().numpy()
                    for data_ in data
                ]  # is cpu needed?
            return data.cpu().detach().numpy()

    def predict_reconstruction(self, *data):
        dataset = MultiviewDataModule.dataset(*data, labels=None)
        generator = DataLoader(dataset, batch_size=self.cfg.datamodule.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_idx, (local_batch) in enumerate(generator):
                local_batch = [
                    local_batch_.to(self.device) for local_batch_ in local_batch
                ]
                if self.variational:
                    mu, logvar = self.encode(local_batch)
                    z = self.reparameterise(mu, logvar)
                else:
                    z = self.encode(local_batch)
                if self.sparse:
                    z = self.apply_threshold(z)
                x_recon = self.decode(z)
                if batch_idx == 0:
                    x_reconstruction = self.process_output(x_recon)
                else:

                    x_reconstruction = self.process_output(
                        x_recon, pred=x_reconstruction
                    )
            return x_reconstruction

    def predict_labels(self, *data):
        dataset = MultiviewDataModule.dataset(*data, labels=None)
        generator = DataLoader(dataset, batch_size=self.cfg.datamodule.batch_size, shuffle=False)
        with torch.no_grad():
            for batch_idx, (local_batch) in enumerate(generator):
                local_batch = [
                    local_batch_.to(self.device) for local_batch_ in local_batch
                ]
                mu, logvar = self.encode(local_batch)
                z = self.reparameterise(mu, logvar)
                output = self.classify(z)
                pred = [torch.argmax(output_, dim=1) for output_ in output]
                if batch_idx == 0:
                    predictions = self.process_output(pred, data_type="prediction")
                else:
                    predictions = self.process_output(
                        pred, pred=predictions, data_type="prediction"
                    )
            return predictions

    def _step(self, batch, batch_idx, stage):
        fwd_return = self.forward(batch)
        loss = self.loss_function(batch, fwd_return)     
        for loss_n, loss_val in loss.items():
            self.log(
                f"{stage}_{loss_n}", loss_val, on_epoch=True, prog_bar=True, logger=True
            ) 
        return loss['loss']         

    def training_step(self, batch, batch_idx, optimizer_idx):
        return self._step(batch, batch_idx, stage='train')

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, stage='val')

    def on_train_end(self):
        self.trainer.save_checkpoint(join(self.cfg.out_dir, "model.ckpt"))
        torch.save(self, join(self.cfg.out_dir, "model.pkl"))

class BaseModelAAE(BaseModel):
    def __init__(self):
        super().__init__()

    def validate_batch(self, local_batch):
        with torch.no_grad():
            self.eval()
            fwd_return = self.forward_recon(local_batch)
            loss_recon = self.recon_loss(self, local_batch, fwd_return)
            fwd_return = self.forward_discrim(local_batch)
            loss_disc = self.discriminator_loss(self, fwd_return)
            fwd_return = self.forward_gen(local_batch)
            loss_gen = self.generator_loss(self, fwd_return)
            loss_total = loss_recon + loss_disc + loss_gen
            loss = {
                "total": loss_total,
                "recon": loss_recon,
                "disc": loss_disc,
                "gen": loss_gen,
            }
        return loss

    def optimise_batch(self, local_batch):
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
        if self.wasserstein:
            for p in self.discriminator.parameters():
                p.data.clamp_(-0.01, 0.01)
        fwd_return = self.forward_gen(local_batch)
        loss_gen = self.generator_loss(fwd_return)
        [optimizer.zero_grad() for optimizer in gen_opt]
        self.manual_backward(loss_gen)
        [optimizer.step() for optimizer in gen_opt]
        loss_total = loss_recon + loss_disc + loss_gen
        loss = {
            "total": loss_total,
            "recon": loss_recon,
            "disc": loss_disc,
            "gen": loss_gen,
        }
        return loss
