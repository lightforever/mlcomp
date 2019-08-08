from catalyst.dl import registry
from catalyst.contrib.models.segmentation import (
    Unet, ResnetLinknet, MobileUnet, ResnetUnet, ResnetFPNUnet, ResnetPSPnet,
    FPNUnet, Linknet, PSPnet
)

from mlcomp.contrib.model import Pretrained
from mlcomp.contrib.criterion import RingLoss
from mlcomp.contrib.catalyst.callbacks.inference import InferBestCallback
from mlcomp.contrib.catalyst.optim import OneCycleCosineAnnealLR


def register():
    registry.Model(Pretrained)

    registry.Criterion(RingLoss)

    registry.Callback(InferBestCallback)

    registry.Scheduler(OneCycleCosineAnnealLR)

    # segmentation
    registry.Model(Unet)
    registry.Model(ResnetLinknet)
    registry.Model(MobileUnet)
    registry.Model(ResnetUnet)
    registry.Model(ResnetFPNUnet)
    registry.Model(ResnetPSPnet)
    registry.Model(FPNUnet)
    registry.Model(Linknet)
    registry.Model(PSPnet)


__all__ = ['register']
