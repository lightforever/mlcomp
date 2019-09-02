from torch import nn

import segmentation_models_pytorch as smb


class SegmentationModelPytorch(nn.Module):
    def __init__(self, arch: str, encoder: str, num_classes: int = 1,
                 encoder_weights: str = 'imagenet', **kwargs):
        super().__init__()

        model = getattr(smb, arch)
        self.model = model(encoder_name=encoder, classes=num_classes,
                           encoder_weights=encoder_weights, **kwargs)

    def forward(self, x):
        return self.model.forward(x)


__all__ = ['SegmentationModelPytorch']
