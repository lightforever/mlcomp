from catalyst.dl import registry
from catalyst.contrib.models.segmentation import (
    Unet, ResnetLinknet, MobileUnet, ResnetUnet, ResnetFPNUnet, ResnetPSPnet,
    FPNUnet, Linknet, PSPnet,
    ResNetLinknet)

from mlcomp.contrib.criterion import RingLoss
from mlcomp.contrib.catalyst.callbacks.inference import InferBestCallback
from mlcomp.contrib.catalyst.optim import OneCycleCosineAnnealLR
from mlcomp.contrib.model.segmentation_model_pytorch import \
            SegmentationModelPytorch
from mlcomp.contrib.model import Pretrained


def register():
    registry.Criterion(RingLoss)

    registry.Callback(InferBestCallback)

    registry.Scheduler(OneCycleCosineAnnealLR)

    # classification
    registry.Model(Pretrained)

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
    registry.Model(ResNetLinknet)

    registry.Model(SegmentationModelPytorch)


__all__ = ['register']
