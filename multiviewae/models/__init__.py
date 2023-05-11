from .ae import *
from .aae import *
from .joint_aae import *
from .waae import *
from .mcvae import *
from .mvae import *
from .mmvae import *
from .mvtcae import *
from .dvcca import *
from .me_mvae import *
from .jmvae import *
from .mopoevae import *
from .mmJSD import *
from .weighted_mvae import *
__all__ = ["AE", "AAE", "jointAAE", "wAAE", "mcVAE", "mVAE", "JMVAE", "me_mVAE", "mmVAE", "mvtCAE", "DVCCA", "MoPoEVAE", "mmJSD", "weighted_mVAE"]
classes = __all__ 
#TODO: use constants instead