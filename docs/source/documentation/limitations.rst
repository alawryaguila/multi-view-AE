Limitations and user implementations
====================================

There are a number of limitations and allowed parameter combinations within the ``multiviewAE`` framework. These restrictions are set in the ``multiviewae/base/validation.py`` file and will need to be updated should the user wish to add their own implementations.
Allowed parameter types are also set in the ``multiviewae/base/validation.py`` file.

Limitations and allowed parameter combinations
----------------------------------------------

Distribution classes
^^^^^^^^^^^^^^^^^^^^
It should be noted that currently the multivariate normal class, ``multiviewae.base.distributions.MultivariateNormal``, implements a multivariate gaussian with a diagonal covariance matrix.
Further work will involve implementing a multivariate normal class where the off-diagonal elements of the covariance matrix can be specified or learnt.

Encoder distribution
^^^^^^^^^^^^^^^^^^^^
The ``multiviewae.base.distributions.Default`` class must be used for the vanilla autoencoder and adversarial autoencoder implementations where no distribution is specified.

Either the ``multiviewae.base.distributions.Normal`` or ``multiviewae.base.distributions.MultivariateNormal`` classes must be used for variational models.

For adversarial autoencoders with gaussian posterior, i.e. gaussian encoding distributions, the ``multiviewae.base.distributions.Normal`` or ``multiviewae.base.distributions.MultivariateNormal`` classes can be used 
if their are coupled with a variational encoder architecture, e.g. ``multiviewae.architectures.mlp.VariationalEncoder``.

Encoder and prior distribution combinations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Currently, the encoder distribution must be the same as the prior distribution.

Models which support CNN architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Many of the autoencoder models in the ``multiviewAE`` support CNN encoder and decoder network architectures. The  ``JMVAE`` and  ``MMVAE`` models do not currently support these architectures. 
This will be addressed in further work.   

Adding user designed classes
----------------------------
With the exception of network architectures, for the user to use their implemented class, they must have access to the source code and the class must be added to the supported classes in the 
``multiviewae/base/validation.py`` file.

User designed network architectures
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
User designed MLP network classes must be implemented in a ``mlp.py`` file and named one of; ``Encoder``, ``VariationalEncoder``, ``Decoder``, and ``VariationalDecoder`` depending on the network type.
CNN network classes must be implemented in a ``cnn.py`` file and named one of; ``Encoder``, ``VariationalEncoder``, and ``Decoder`` depending on the network type.

Networks must except and return the same parameters as the respective ``multiviewAE`` counterpart. 
For example, variational encoder networks must return ``mu`` and ``logvar`` in the form of a ``Torch.tensor``. 
Please see the :ref:`Architectures` section for information on input and output parameters of encoder and decoder networks. 

Implemented classes must have a ``forward`` method.

User designed distributions
^^^^^^^^^^^^^^^^^^^^^^^^^^^
User designed distributions must implement a ``log_likelihood`` and ``_sample`` method.

User designed DataModules and Datasets
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
The ``multiviewAE`` library does not currently allow for a user designed ``DataModule`` or ``Dataset``. This functionality will be implemented in the future.
