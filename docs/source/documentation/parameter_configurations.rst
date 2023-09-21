Parameter settings and configurations
=====================================

This file containing information on parameters settings in the configuration files, how to set them, and allowed combinations of parameters.

How to specify the yaml configuration file
------------------------------------------

Most parameters (with the exception of those discussed in the User Guide) are set using configuration yaml files. There are some example yaml files given in the ``multi-view-AE/tests/user_config/`` folder. To specify a configuration file, the user must specify the absolute or relative path to the yaml file when initialising the relevant model:

.. code-block:: python

   from multiviewae import mVAE
   
   mvae = mVAE(cfg='./config_folder/test_config.yaml',
        input_dim=[20, 20],
        z_dim=2)


If no configuration file is specified, the default configuration for that model class is used. These can be found in the ``multi-view-AE/multiviewae/configs/model_type/`` folder.

Configuration file structure
----------------------------

The configuration file has the following model parameter groupings which can be edited by the user. 

Model
^^^^^

The global model parameter settings. 

.. code-block:: yaml

        model:
          save_model: True

          seed_everything: True
          seed: 42

          z_dim: 5
          learning_rate: 0.001 

          sparse: False

There are also a number of model specific parameters which are set in the yaml files in the ``multi-view-AE/multiviewae/configs/model_type/`` folder.

Datamodule
^^^^^^^^^^

The parameters for the PyTorch data module class to access and process the data.

.. code-block:: yaml

        datamodule:
          _target_: multiviewae.base.dataloaders.MultiviewDataModule

          batch_size: null
          is_validate: True

          train_size: 0.9

When the batch size is set to ``null``  the full batch is used for training at each epoch. 

MLP Encoder
^^^^^^^^^^^

The encoder function parameters. The default encoder function is a MLP encoder network:

.. code-block:: yaml

        encoder:  
          default:
              _target_: multiviewae.architectures.mlp.Encoder

              hidden_layer_dim: []
              bias: True 
              non_linear: False

              enc_dist:
                _target_: multiviewae.base.distributions.Default

The ``encoder._target_`` parameter specifies the encoder function class of which the in-built options include: ``multiviewae.architectures.mlp.Encoder`` and ``multiviewae.architectures.mlp.VariationalEncoder``.

The ``encoder.enc_dist._target_`` parameter specifies the encoding distribution class of which the in-built options include: ``multiviewae.base.distributions.Default``, ``multiviewae.base.distributions.Normal`` and ``multiviewae.base.distributions.MultivariateNormal``. The ``multiviewae.base.distributions.Default`` class is used for the vanilla autoencoder and adversarial autoencoder implementations where no distribution is specified.

The user can specify separate parameters for the encoder network of each view. For example:

.. code-block:: yaml

        encoder:  
          enc0:
              _target_: multiviewae.architectures.mlp.Encoder

              hidden_layer_dim: [12, 6]
              bias: True
              non_linear: False

              enc_dist:
                _target_: multiviewae.base.distributions.Default
          enc1:
              _target_: multiviewae.architectures.mlp.Encoder

              hidden_layer_dim: [50, 6]
              bias: True
              non_linear: True

              enc_dist:
                _target_: multiviewae.base.distributions.Default

where ``enc0`` and ``enc1`` provide the parameters for view 0 encoder and view 1 encoder respectively. If no view specific parameters are provided, the default network parameters are used.

**NOTE:** The ``default`` encoder parameters are used for joint encoding distributions.

CNN Encoder
^^^^^^^^^^^

Alternatively, the user can specify a CNN architecture by setting the ``encoder._target_`` parameter:

.. code-block:: yaml

        encoder:
          default:
              _target_: multiviewae.architectures.cnn.Encoder

              layer0:
                layer: Conv2d
                in_channels: 1
                out_channels: 8
                kernel_size: 4
                stride: 2
                padding: 1

              layer1:
                layer: Conv2d
                in_channels: 8
                out_channels: 16
                kernel_size: 4
                stride: 2
                padding: 1

              layer2:
                layer: Conv2d
                in_channels: 16
                out_channels: 32
                kernel_size: 4
                stride: 2
                padding: 1

              layer3:
                layer: Conv2d
                in_channels: 32
                out_channels: 64
                kernel_size: 4
                stride: 2
                padding: 0

              layer5:
                layer: AdaptiveAvgPool2d
                output_size: 1

              layer6:
                layer: Flatten
                start_dim: 1

              layer7:
                layer: Linear
                in_features: 64
                out_features: 128

              bias: True
              non_linear: False

              enc_dist:
                _target_: multiviewae.base.distributions.Default

In-built options include: ``multiviewae.architectures.cnn.Encoder`` and ``multiviewae.architectures.cnn.VariationalEncoder``. As with the MLP architectures, the user can chose to set view specific parameters.
Each layer can be ``torch.nn`` ``Conv2d`` layers or any suitable 2D pooling or padding layers.

**NOTE:** The user is responsible for ensuring that the CNN encoder and decoder network architectures are compatible and create an output tensor of the correct dimensionality.

MLP Decoder
^^^^^^^^^^^

The decoder function parameters. The default decoder function is a MLP decoder network:

.. code-block:: yaml

        decoder:
          default:
              _target_: multiviewae.architectures.mlp.Decoder

              hidden_layer_dim: []
              bias: True 
              non_linear: False

              dec_dist:
                _target_: multiviewae.base.distributions.Default
 
The ``decoder._target_`` parameter specifies the encoder function class of which the in-built options include: ``multiviewae.architectures.mlp.Decoder`` and ``multiviewae.models.layers.VariationalDecoder``.

The ``decoder.dec_dist._target_`` parameter specifies the decoding distribution class of which the in-built options include: ``multiviewae.base.distributions.Default``, ``multiviewae.base.distributions.Normal``, ``multiviewae.base.distributions.MultivariateNormal``, ``multiviewae.base.distributions.Laplace`` and ``multiviewae.base.distributions.Bernoulli``. The ``multiviewae.base.distributions.Default`` class is used for the vanilla autoencoder and adversarial autoencoder implementations where no distribution is specified.

The user can specify separate parameters for the encoder network of each view. For example:

.. code-block:: yaml

        decoder:  
          dec0:
              _target_: multiviewae.architectures.mlp.Encoder

              hidden_layer_dim: [6, 12]
              bias: True
              non_linear: False

              dec_dist:
                _target_: multiviewae.base.distributions.Default
          dec1:
              _target_: multiviewae.architectures.mlp.Encoder

              hidden_layer_dim: [6, 50]
              bias: True
              non_linear: True

              dec_dist:
                _target_: multiviewae.base.distributions.Default

where ``enc0`` and ``enc1`` provide the parameters for view 0 encoder and view 1 encoder respectively. If no view specific parameters are provided, the default network parameters are used.

CNN Decoder
^^^^^^^^^^^

Alternatively, the user can specify a CNN architecture by setting the ``encoder._target_`` parameter:

.. code-block:: yaml

        decoder:
          default:
              _target_: multiviewae.architectures.cnn.Decoder

              layer0: 
                layer: Linear
                out_features: 128

              layer1:
                layer: Linear
                in_features: 128
                out_features: 64

              layer2:
                layer: Unflatten
                dim: 1
                unflattened_size: [64, 1, 1]  

              layer3:
                layer: ConvTranspose2d
                in_channels: 64
                out_channels: 32
                kernel_size: 4
                stride: 2
                padding: 0

              layer4:
                layer: ConvTranspose2d
                in_channels: 32
                out_channels: 16
                kernel_size: 4
                stride: 2
                padding: 1

              layer5:
                layer: ConvTranspose2d
                in_channels: 16
                out_channels: 8
                kernel_size: 4
                stride: 2
                padding: 1

              layer6:
                layer: ConvTranspose2d
                in_channels: 8
                out_channels: 1
                kernel_size: 4
                stride: 2
                padding: 1

              bias: True
              non_linear: False

              dec_dist:
                _target_: multiviewae.base.distributions.Default

**NOTE:** The user is responsible for ensuring that the CNN encoder and decoder network architectures are compatible and create an output tensor of the correct dimensionality.

Prior
^^^^^

The parameters of the prior distribution for variational models. 

.. code-block:: yaml

        prior:
          _target_: multiviewae.base.distributions.Normal
          loc: 0
          scale: 1

The prior can take the form of a univariate gaussian, ``multiviewae.base.distributions.Normal``, or multivariate gaussian, ``multiviewae.base.distributions.MultivariateNormal``,  with diagonal covariance matrix with variances given by the ``scale`` parameter.

Trainer
^^^^^^^

The parameters for the PyTorch trainer. Please see the PyTorch Lightning documentation for more information on the parameter settings.

.. code-block:: yaml

        trainer:
          _target_: pytorch_lightning.Trainer

          accelerator: "auto"

          max_epochs: 10

          deterministic: false
          log_every_n_steps: 2

Callbacks
^^^^^^^^^

Parameters for the PyTorchLightning callbacks. Please see the PyTorch Lightning documentation for more information on the parameter settings.

.. code-block:: yaml

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

Only the ``model_checkpoint`` and ``early_stopping`` callbacks are used in the ``multi-view-AE`` library. However for more callback options, please refer to the PyTorch Lightning documentation.

Logger
^^^^^^

The parameters of the logger file. 

.. code-block:: yaml

        logger:
          _target_: pytorch_lightning.loggers.tensorboard.TensorBoardLogger

          save_dir: ${out_dir}/logs

In the ``multi-view-AE`` we use TensorBoard for logging. However, the user is free to use whichever logging framework their prefer.
**NOTE:** other logging frameworks have not been tested. 

Changing parameter settings
---------------------------

Only the grouping header, sub header and the parameters the user wishes to change need to be specified in the users yaml file. The default model parameters are used for the remaining parameters. For example, to change the number of hidden layers for the encoder and decoder networks the user can use the following yaml file:

.. code-block:: yaml

        encoder:
          hidden_layer_dim: [10, 5]  

        decoder:
          hidden_layer_dim: [10, 5] 


**NOTE:** An exception to this rule are the Pytorch callbacks where all the parameters for the relevant callback must be specified again in the user configuration file. For example to change the early stopping patience to ``100`` of the following callback:

.. code-block:: yaml

        callbacks:
          early_stopping:
            _target_: pytorch_lightning.callbacks.EarlyStopping
            monitor: "val_loss"
            mode: "min"
            patience: 50
            min_delta: 0.001
            verbose: True

The user must add the following section to their yaml file:

.. code-block:: yaml

        callbacks:
          early_stopping:
            _target_: pytorch_lightning.callbacks.EarlyStopping
            monitor: "val_loss"
            mode: "min"
            patience: 100
            min_delta: 0.001
            verbose: True


Target classes
--------------

There are a number of model classes specified in the configuration file, namely; the encoder and decoder functions, the encoder, decoder, and prior distributions for variational models, and the discriminator function for adversarial models. There are a number of existing classes built into the ``multi-view-AE`` framework for the user to chose from. Alternatively, the user can use their own classes and specify them in the yaml file:

.. code-block:: yaml

        encoder:
          _target_: encoder_folder.user_encoder

        decoder:
          _target_: decoder_folder.user_decoder

However, for these classes to work with the ``multi-view-AE`` framework, user class implementations must follow the same structure as existing classes. For example, an ``encoder`` implementation must have a ``forward`` method.

Allowed parameter combinations
------------------------------

Some parameter combinations are not compatible in the ``multi-view-AE`` framework. If an incorrect parameter combination is given in the configuration file, either a warning or error is raised depending on whether the parameter choices can be ignored or would impede the model from functioning correctly.
