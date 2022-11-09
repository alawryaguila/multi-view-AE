from schema import Schema, And, Or, Optional, SchemaError, Regex

SUPPORTED_ENCODERS = [
            "multiae.architectures.mlp.Encoder",
            "multiae.architectures.mlp.VariationalEncoder",
            "multiae.architectures.cnn.Encoder",
            "multiae.architectures.cnn.VariationalEncoder"
        ]

SUPPORTED_DECODERS = [
            "multiae.architectures.mlp.Decoder",
            "multiae.architectures.mlp.VariationalDecoder",
            "multiae.architectures.cnn.Decoder"
        ]

SUPPORTED_DISCRIMINATORS = [
            "multiae.architectures.mlp.Discriminator"
        ]

SUPPORTED_DISTRIBUTIONS = [
            "multiae.base.distributions.Default",
            "multiae.base.distributions.Normal",
            "multiae.base.distributions.MultivariateNormal",
            "multiae.base.distributions.Bernoulli", 
            "multiae.base.distributions.Laplace"
        ]

SUPPORTED_JOIN = [
            "PoE",
            "Mean"
        ]

UNSUPPORTED_ENC_DIST = [
            "multiae.base.distributions.Bernoulli"
        ]

UNSUPPORTED_PRIOR_DIST = [
            "multiae.base.distributions.Default",
            "multiae.base.distributions.Bernoulli"
        ]

def return_or(params=[], msg="invalid"):
    assert(len(params) > 0)

    p = ', '.join('"{0}"'.format(str(w)) for w in params)
    return f"Or({p}, error='{msg}')"

def list_sub(a, b):
    return list(set(a) - set(b))

config_schema = Schema({
    "model": {
        "use_GPU": bool,
        "save_model": bool,
        "seed_everything": bool,
        "seed": And(int, lambda x: 0 <= x <= 4294967295),
        "z_dim": int,
        "learning_rate": And(float, lambda x: 0 < x < 1),
        "sparse": bool,
        "threshold": Or(And(float, lambda x: 0 < x < 1), 0),
        Optional("eps"): And(float, lambda x: 0 < x <= 1e-10),
        Optional("beta"): And(Or(int, float), lambda x: x > 0),
        Optional("K"): And(int, lambda x: x >= 1),
        Optional("alpha"): And(float, lambda x: x > 0),
        Optional("private"): bool,
        Optional("join_type"): eval(return_or(params=SUPPORTED_JOIN,
                        msg="model.join_type: unsupported or invalid join type"))
    },
    "datamodule": {
        "_target_": "multiae.base.dataloaders.MultiviewDataModule",    #TODO: how about user-defined classes?
        "batch_size": Or(And(int, lambda x: x > 0), None),
        "is_validate": bool,
        "train_size": And(float, lambda x: 0 < x < 1)
    },
    "encoder": {
        "default": {
            "_target_" : eval(return_or(params=SUPPORTED_ENCODERS,
                            msg="encoder._target_: unsupported or invalid encoder")),
            Optional("hidden_layer_dim"): [And(int, lambda x: x > 0)],
            Optional(Regex(r'^layer\d$')) : {
                "layer": str, # TODO: fix this. should be specific torch.nn layers
                # TODO: how about parameters of each layer? will raise error anyway...
            },
            "bias": bool,
            "non_linear": bool,
            "enc_dist": {
                    "_target_": eval(return_or(params=list_sub(SUPPORTED_DISTRIBUTIONS, UNSUPPORTED_ENC_DIST),
                            msg="encoder.enc_dist._target_: unsupported or invalid encoder distribution"))
            }
        },
        Optional(Regex(r'^enc\d$')) : {
            "_target_" : eval(return_or(params=SUPPORTED_ENCODERS,
                            msg="encoder._target_: unsupported or invalid encoder")),
            Optional("hidden_layer_dim"): [And(int, lambda x: x > 0)],
            Optional(Regex(r'^layer\d$')) : {
                "layer": str, # TODO: fix this. should be specific torch.nn layers
                # TODO: how about parameters of each layer? will raise error anyway...
            },
            "bias": bool,
            "non_linear": bool,
            "enc_dist": {
                    "_target_": eval(return_or(params=list_sub(SUPPORTED_DISTRIBUTIONS, UNSUPPORTED_ENC_DIST),
                            msg="encoder.enc_dist._target_: unsupported or invalid encoder distribution"))
            }
        }
    },
    "decoder": {
        "default": {
            "_target_" : eval(return_or(params=SUPPORTED_DECODERS,
                            msg="decoder._target_: unsupported or invalid decoder")),
            Optional("hidden_layer_dim"): [And(int, lambda x: x > 0)],
            Optional(Regex(r'^layer\d$')) : {
                "layer": str, # TODO: fix this. should be specific torch.nn layers
                # TODO: how about parameters of each layer? will raise error anyway...
            },
            "bias": bool,
            "non_linear": bool,
            Optional("init_logvar"): Or(int, float),
            "dec_dist": {
                    "_target_": eval(return_or(params=SUPPORTED_DISTRIBUTIONS,
                            msg="decoder.dec_dist._target_: unsupported or invalid decoder distribution"))
            }
        },
        Optional(Regex(r'^dec\d$')) : {
            "_target_" : eval(return_or(params=SUPPORTED_DECODERS,
                            msg="decoder._target_: unsupported or invalid decoder")),
            Optional("hidden_layer_dim"): [And(int, lambda x: x > 0)],
            Optional(Regex(r'^layer\d$')) : {
                "layer": str, # TODO: fix this. should be specific torch.nn layers
                # TODO: how about parameters of each layer? will raise error anyway...
            },
            "bias": bool,
            "non_linear": bool,
            Optional("init_logvar"): Or(int, float),
            "dec_dist": {
                    "_target_": eval(return_or(params=SUPPORTED_DISTRIBUTIONS,
                            msg="decoder.dec_dist._target_: unsupported or invalid decoder distribution"))
            }
        }
    },
    Optional("discriminator"): {
        "_target_" : eval(return_or(params=SUPPORTED_DISCRIMINATORS,
                        msg="discriminator._target_: unsupported or invalid discriminator")),
        "hidden_layer_dim": [And(int, lambda x: x > 0)],
        "bias": bool,
        "non_linear": bool,
        "dropout_threshold": Or(0, And(float, lambda x: 0 < x < 1))
    },
    "prior": {
       "_target_" : eval(return_or(params=list_sub(SUPPORTED_DISTRIBUTIONS, UNSUPPORTED_PRIOR_DIST),
                        msg="prior._target_: unsupported or invalid prior")),
       "loc":  Or(Or(int, float), [Or(int, float)]),
       "scale": Or(And(Or(int, float), lambda x: x > 0), [And(Or(int, float), lambda x: x > 0)])
    },
    "trainer": {
       "_target_" : "pytorch_lightning.Trainer",
       "gpus": And(int, lambda x: x >= 0),
       "max_epochs": And(int, lambda x: x > 0),
       "deterministic": bool,
       "log_every_n_steps": And(int, lambda x: x > 0),
       "resume_from_checkpoint": Or(str, None) # TODO: check valid location?
    },
    "callbacks": {
        "model_checkpoint": {   #TODO: how about other params?
           "_target_" : "pytorch_lightning.callbacks.ModelCheckpoint",
           "monitor": Or("train_loss", "val_loss"), # see training_step() and validation_step()
           "mode": Or("min", "max"),
           "save_last": bool,
           "dirpath": str   # TODO: check valid location?
        },
        "early_stopping": { #TODO: how about other params?
           "_target_" : "pytorch_lightning.callbacks.EarlyStopping",
           "monitor": Or("train_loss", "val_loss"), # see training_step() and validation_step()
           "mode": Or("min", "max"),
           "patience": And(int, lambda x: x > 0),
           "min_delta": float,
           "verbose": bool
        }
    },
    "logger": {
       "_target_" : "pytorch_lightning.loggers.tensorboard.TensorBoardLogger",
       "save_dir": str   # TODO: check valid location?
    }
}, ignore_extra_keys=True)
