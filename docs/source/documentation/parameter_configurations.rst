Parameter settings and configurations
===============

This file containing information on parameters settings in the configuration files, how to set them, and allowed combinations of parameters.

How to specify the yaml configuration file
---------------------------------

Most parameters (with the exception of those discussed in the User Guide) are set using configuration yaml files. There are some example yaml files given in the ```multiviewAE/tests/user_config/``` folder. To specify a configuration file, the user must specify the absolute or relative path to the yaml file when initialising the relevant model:

.. code-block:: python

   from multiae import mVAE
   
   mvae = mVAE(cfg='./config_folder/test_config.yaml',
        input_dim=[20, 20],
        z_dim=2)


If no configuration file is specified, the default configuration for that model class is used. These can be found in the ``multiviewAE/multiae/configs/model_type/`` folder.

Configuration file structure
--------------------------------

The configuration file has the following model parameter groupings which can be edited by the user. 

Model
^^^^^

The global model parameter settings. 

.. code-block:: python

        model:
          use_GPU: False
          save_model: True

          seed_everything: True
          seed: 42

          z_dim: 5
          learning_rate: 0.001 

          sparse: False

There are also a number of model specific parameters which are set in the yaml files in the ``multiviewAE/multiae/configs/model_type/`` folder.

Datamodule
^^^^^

The parameters for the PyTorch data module class to access and process the data.

.. code-block:: python

        datamodule:
          _target_: multiae.base.dataloaders.MultiviewDataModule

          batch_size: null
          is_validate: True

          train_size: 0.9

When the batch size is set to ``null``  the full batch is used for training at each epoch. 

Encoder
^^^^^

The encoder function parameters.

.. code-block:: python

        encoder:
          _target_: multiae.models.layers.Encoder

          hidden_layer_dim: []  
          bias: True
          non_linear: False

          enc_dist:
            _target_: multiae.base.distributions.Default
 
The ``encoder._target_`` parameter specifies the encoder function class of which the in-built options include: ``multiae.models.layers.Encoder`` and ``multiae.models.layers.VariationalEncoder``.

The ``encoder.enc_dist._target_`` parameter specifies the encoding distribution class of which the in-built options include: ``multiae.base.distributions.Default``, ``multiae.base.distributions.Normal`` and ``multiae.base.distributions.MultivariateNormal``. The ``multiae.base.distributions.Default`` class is used for the vanilla autoencoder and adversarial autoencoder implementations where no distribution is specified.

Decoder
^^^^^

The decoder function parameters.

.. code-block:: python

        decoder:
          _target_: multiae.models.layers.Decoder

          hidden_layer_dim: []
          bias: True
          non_linear: False

          dec_dist:
            _target_: multiae.base.distributions.Default
 
The ``decoder._target_`` parameter specifies the encoder function class of which the in-built options include: ``multiae.models.layers.Decoder`` and ``multiae.models.layers.VariationalDecoder``.

The ``decoder.dec_dist._target_`` parameter specifies the decoding distribution class of which the in-built options include: ``multiae.base.distributions.Default``, ``multiae.base.distributions.Normal``, ``multiae.base.distributions.MultivariateNormal`` and ``multiae.base.distributions.Bernoulli``. The ``multiae.base.distributions.Default`` class is used for the vanilla autoencoder and adversarial autoencoder implementations where no distribution is specified.

**NOTE:** The order of the layer dimensions in ``hidden_layer_dim`` is flipped by the model. Such that ``hidden_layer_dim=[10, 5]`` indicates a decoder network architecture:

.. code-block:: python

        z_dim --> 5 --> 10 --> input_dim

Prior
^^^^^

The parameters of the prior distribution for variational models. 

.. code-block:: python

        prior:
          _target_: multiae.base.distributions.Normal
          loc: 0
          scale: 1

The prior can take the form of a univariate gaussian, ``multiae.base.distributions.Normal``, or multivariate gaussian, ``multiae.base.distributions.MultivariateNormal``,  with diagonal covariance matrix with variances given by the ``scale`` parameter.

Trainer
^^^^^

The parameters for the PyTorch trainer. 

.. code-block:: python

        trainer:
          _target_: pytorch_lightning.Trainer

          gpus: 0

          max_epochs: 10

          deterministic: false
          log_every_n_steps: 2

          resume_from_checkpoint: ${out_dir}/last.ckpt  

Callbacks
^^^^^

Parameters for the PyTorchLightning callbacks.

.. code-block:: python

        callbacks:
          model_checkpoint:
            _target_: pytorch_lightning.callbacks.ModelCheckpoint
            monitor: "val_loss"
            mode: "min"
            save_last: True
            dirpath: ${out_dir}

          early_stopping:
            _target_: pytorch_lightning.callbacks.EarlyStopping
            monitor: "val_loss"
            mode: "min"
            patience: 50
            min_delta: 0.001
            verbose: True

Only the ``model_checkpoint`` and ``early_stopping`` callbacks are used in the ``multiviewAE`` library. However for more callback options, please refer to the PyTorch Lightning documentation.

Logger
^^^^^

The parameters of the logger file. 

.. code-block:: python

        logger:
          _target_: pytorch_lightning.loggers.tensorboard.TensorBoardLogger

          save_dir: ${out_dir}/logs

In the ``multiviewAE`` we use TensorBoard for logging. However, the user is free to use whichever logging framework their prefer.

Changing parameter settings
--------------------------------

Only the grouping header, sub header and the parameters the user wishes to change need to be specified in the users yaml file. The default model parameters are used for the remaining parameters. For example, to change the number of hidden layers for the encoder and decoder networks the user can use the following yaml file:

.. code-block:: python

        encoder:
          hidden_layer_dim: [10, 5]  

        decoder:
          hidden_layer_dim: [10, 5] 


**NOTE:** An exception to this rule are the Pytorch callbacks where all the parameters for the relevant callback must be specified again in the user configuration file. For example to change the early stopping patience to ``100`` of the following callback:

.. code-block:: python

        callbacks:
          early_stopping:
            _target_: pytorch_lightning.callbacks.EarlyStopping
            monitor: "val_loss"
            mode: "min"
            patience: 50
            min_delta: 0.001
            verbose: True

The user must add the following section to their yaml file:

.. code-block:: python

        callbacks:
          early_stopping:
            _target_: pytorch_lightning.callbacks.EarlyStopping
            monitor: "val_loss"
            mode: "min"
            patience: 100
            min_delta: 0.001
            verbose: True


Target classes
--------------------------------

There are a number of model classes specified in the configuration file, namely; the encoder and decoder functions, the encoder, decoder, and prior distributions for variational models, and the discriminator function for adversarial models. There are a number of existing classes built into the ``multiviewAE`` framework for the user to chose from. Alternatively, the user can use their own classes and specify them in the yaml file:

.. code-block:: python

        encoder:
          _target_: encoder_folder.user_encoder

        decoder:
          _target_: decoder_folder.user_decoder

However, for these classes to work with the ``multiviewAE`` framework, user class implementations must follow the same structure as existing classes. For example, an ``encoder`` implementation must have a ``forward`` method.

Allowed parameter combinations
--------------------------------

Some parameter combinations are not compatible in the ``multiviewAE`` framework. If an incorrect parameter combination is given in the configuration file, either a warning or error is raised depending on whether the parameter choices can be ignored or would impede the model from functioning correctly.
