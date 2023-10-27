Limitations and user implementations
====================================

There are a number of limitations and allowed parameter combinations within the ``multi-view-AE`` framework. These restrictions are set in the ``multiviewae/base/validation.py`` file and will need to be updated should the user wish to add their own implementations.
Allowed parameter types are also set in the ``multiviewae/base/validation.py`` file.

Limitations and allowed parameter combinations
----------------------------------------------

Distribution classes
^^^^^^^^^^^^^^^^^^^^
It should be noted that currently the multivariate normal class, ``multiviewae.base.distributions.MultivariateNormal``, implements a multivariate gaussian with a diagonal covariance matrix.
Further work will involve implementing a multivariate normal class where the off-diagonal elements of the covariance matrix can be specified or learnt.

Encoder distribution
^^^^^^^^^^^^^^^^^^^^
The ``multiviewae.base.distributions.Default_dist`` class must be used for the vanilla autoencoder and adversarial autoencoder implementations where no distribution is specified.

Either the ``multiviewae.base.distributions.Normal`` or ``multiviewae.base.distributions.MultivariateNormal`` classes must be used for variational models.

For adversarial autoencoders with gaussian posterior, i.e. gaussian encoding distributions, the ``multiviewae.base.distributions.Normal`` or ``multiviewae.base.distributions.MultivariateNormal`` classes can be used 
if their are coupled with a variational encoder architecture, e.g. ``multiviewae.architectures.mlp.VariationalEncoder``.

Encoder and prior distribution combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, the encoder distribution must be the same as the prior distribution.

Models which support CNN architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many of the autoencoder models in the ``multi-view-AE`` support CNN encoder and decoder network architectures. The  ``JMVAE`` and  ``MMVAE`` models do not currently support these architectures. 
This will be addressed in further work.   

Adding user designed classes
----------------------------
Users are able to implement their own network architectures, datamodules, datasets and distributions. This should cover the majority of classes the user should wish to edit and offers lots of flexibility when implementing a model. For any other classes, users must have access to the source code and the class must be added to the supported classes in the 
``multiviewae/base/validation.py`` file.

User designed network architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
User designed MLP network classes must be implemented in a ``mlp.py`` file and named one of; ``Encoder``, ``VariationalEncoder``, ``Decoder``, and ``VariationalDecoder`` depending on the network type.
CNN network classes must be implemented in a ``cnn.py`` file and named one of; ``Encoder``, ``VariationalEncoder``, and ``Decoder`` depending on the network type.

Networks must except and return the same parameters as the respective ``multi-view-AE`` counterpart. 
For example, variational encoder networks must return ``mu`` and ``logvar`` in the form of a ``Torch.tensor``. 
Please see the :ref:`Architectures` section for information on input and output parameters of encoder and decoder networks. 

Implemented classes must have a ``forward`` method.

User designed datamodules and datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
User designed datamodules must implement a ``setup`` method, a ``train_dataloader`` method and a ``val_dataloader`` method and must accept the same parameters as the ``multi-view-AE`` counterpart. The datamodule class name must end in ``DataModule``.
User designed datasets must implement a ``__getitem__`` method and must accept the same parameters as the ``multi-view-AE`` counterpart. The dataset class name must end in ``Dataset``.
**NOTE** User implemented datasets must also provide a ``is_path_ds`` parameter to indicate whether the input data is a path to the data for the data to be loaded when the ``__getitem__`` method is called or whether the data is stored in memory.

User designed distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
User designed distributions must implement a ``log_likelihood`` and ``_sample`` method.

