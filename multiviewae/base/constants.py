# constants file
MODEL_AE        = "AE"
MODEL_MAAE       = "mAAE"
MODEL_MWAE      = "mWAE"
MODEL_MCVAE     = "mcVAE"
MODEL_MVAE      = "mVAE"
MODEL_JMVAE     = "JMVAE"
MODEL_MEMVAE    = "me_mVAE"
MODEL_MMVAE     = "mmVAE"
MODEL_MVTCAE    = "mvtCAE"
MODEL_DVCCA     = "DVCCA"
MODEL_MOPOEVAE  = "MoPoEVAE"
MODEL_MMJSD     = "mmJSD"
MODEL_WEIGHTEDMVAE = "weighted_mVAE"
MODEL_VAEBARLOW = "VAE_barlow"
MODEL_AEBARLOW  = "AE_barlow"
MODEL_DMVAE = "DMVAE"
MODEL_WEIGHTEDDMVAE = "weighted_DMVAE"
MODEL_MMVAEPLUS = "mmVAEPlus"

MODELS = [
            MODEL_AE,
            MODEL_MAAE,
            MODEL_MWAE,
            MODEL_MCVAE,
            MODEL_JMVAE,
            MODEL_MVAE,
            MODEL_MEMVAE,
            MODEL_MMVAE,
            MODEL_MVTCAE,
            MODEL_DVCCA,
            MODEL_MOPOEVAE,
            MODEL_MMJSD,
            MODEL_WEIGHTEDMVAE,
            MODEL_VAEBARLOW,
            MODEL_AEBARLOW, 
            MODEL_DMVAE, 
            MODEL_WEIGHTEDDMVAE, 
            MODEL_MMVAEPLUS,
        ]

ADVERSARIAL_MODELS = [
            MODEL_MAAE,
            MODEL_MWAE,
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
            MODEL_MMJSD,
            MODEL_WEIGHTEDMVAE,
            MODEL_VAEBARLOW,
            MODEL_DMVAE,
            MODEL_WEIGHTEDDMVAE,
            MODEL_MMVAEPLUS
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

EPS = 1e-8