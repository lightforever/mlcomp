from catalyst.dl import registry
from catalyst.contrib.models.segmentation import (
    Unet, ResnetLinknet, MobileUnet, ResnetUnet, ResnetFPNUnet, ResnetPSPnet,
    FPNUnet, Linknet, PSPnet,
    ResNetLinknet)

from mlcomp.contrib.criterion import RingLoss
from mlcomp.contrib.catalyst.callbacks.inference import InferBestCallback
from mlcomp.contrib.catalyst.optim import OneCycleCosineAnnealLR


def register():
    registry.Criterion(RingLoss)

    registry.Callback(InferBestCallback)

    registry.Scheduler(OneCycleCosineAnnealLR)

    # classification
    try:
        from mlcomp.contrib.model import Pretrained
        registry.Model(Pretrained)
    except Exception:
        pass

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

    try:
        from mlcomp.contrib.model.segmentation_model_pytorch import \
            SegmentationModelPytorch
        registry.Model(SegmentationModelPytorch)
    except Exception:
        pass


__all__ = ['register']
