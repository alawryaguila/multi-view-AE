# constants file

MODEL_AE        = "AE"
MODEL_AAE       = "AAE"
MODEL_JOINTAAE  = "jointAAE"
MODEL_WAAE      = "wAAE"
MODEL_MCVAE     = "mcVAE"
MODEL_MVAE      = "mVAE"
MODEL_JMVAE     = "JMVAE"
MODEL_MEMVAE    = "me_mVAE"
MODEL_MMVAE     = "mmVAE"
MODEL_MVTCAE    = "mvtCAE"
MODEL_DVCCA     = "DVCCA"
MODEL_MOPOEVAE  = "MoPoEVAE"
MODEL_MMJSD     = "mmJSD"
MODELS = [
            MODEL_AE,
            MODEL_AAE,
            MODEL_JOINTAAE,
            MODEL_WAAE,
            MODEL_MCVAE,
            MODEL_JMVAE,
            MODEL_MVAE,
            MODEL_MEMVAE,
            MODEL_MMVAE,
            MODEL_MVTCAE,
            MODEL_DVCCA,
            MODEL_MOPOEVAE,
            MODEL_MMJSD
        ]

ADVERSARIAL_MODELS = [
            MODEL_AAE,
            MODEL_JOINTAAE,
            MODEL_WAAE,
        ]

VARIATIONAL_MODELS = [
            MODEL_MCVAE,
            MODEL_MVAE,
            MODEL_JMVAE,
            MODEL_MEMVAE,
            MODEL_MMVAE,
            MODEL_MVTCAE,
            MODEL_DVCCA,
            MODEL_MOPOEVAE,
            MODEL_MMJSD
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
            "prior",
            "trainer",
            "callbacks",
            "logger"
        ]
