'''
trainer: function for creating pytorch lightning trainer

'''
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning import Trainer
from os.path import join, exists

def trainer(output_path,
            n_epochs,
            checkpoint_metric_name='val_loss',
            checkpoint_monitor_mode='min',
            early_stopping=True,
            early_stopping_delta=1e-3, 
            early_stopping_patience=10
            ):
    callbacks = []       
    callbacks.append(ModelCheckpoint(monitor=checkpoint_metric_name,
                                        dirpath=output_path,
                                        mode=checkpoint_monitor_mode,
                                        save_last=True
                                        ))
    if early_stopping:
        callbacks.append(EarlyStopping(monitor=checkpoint_metric_name,
                                        min_delta=early_stopping_delta,
                                        patience=early_stopping_patience,
                                        verbose=True,
                                        mode=checkpoint_monitor_mode,))

    logger = pl.loggers.TensorBoardLogger(
            save_dir=join(output_path, 'logs')
    )

    resume_checkpoint = join(output_path, 'last.ckpt')
    if not exists(resume_checkpoint):
        resume_checkpoint = None
    else:
        print('resuming training from checkpoint: ', resume_checkpoint)
    trainer = Trainer(resume_from_checkpoint=resume_checkpoint,
                    max_epochs=n_epochs,
                    logger=logger,
                    callbacks=callbacks)

    return trainer


