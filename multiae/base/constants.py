# constants file

MODEL_AE        = "AE"
MODEL_AAE       = "AAE"
MODEL_JOINTAAE  = "jointAAE"
MODEL_WAAE      = "wAAE"
MODEL_MCVAE     = "mcVAE"
MODEL_MVAE      = "mVAE"
MODEL_MEMVAE    = "me_mVAE"
MODEL_MMVAE     = "mmVAE"
MODEL_MVTCAE    = "mvtCAE"
MODEL_DVCCA     = "DVCCA"

MODELS = [
            MODEL_AE,
            MODEL_AAE,
            MODEL_JOINTAAE,
            MODEL_WAAE,
            MODEL_MCVAE,
            MODEL_MVAE,
            MODEL_MEMVAE,
            MODEL_MMVAE,
            MODEL_MVTCAE,
            MODEL_DVCCA
        ]

VARIATIONAL_MODELS = [
            MODEL_MCVAE,
            MODEL_MVAE,
            MODEL_MEMVAE,
            MODEL_MMVAE,
            MODEL_MVTCAE,
            MODEL_DVCCA
        ]

SPARSE_MODELS = [
            MODEL_MCVAE,
            MODEL_MVAE,
            MODEL_DVCCA
        ]

CONFIG_KEYS = [
            "model",
            "datamodule",
            "encoder",
            "decoder",
            "trainer",
            "callbacks",
            "logger"
        ]
