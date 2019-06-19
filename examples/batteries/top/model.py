import torch.nn as nn
import torch.nn.functional as F
from catalyst.contrib import registry
from catalyst.contrib.models.segmentation import ResNetUnet
from catalyst.contrib.models.segmentation.linknet import Linknet
from catalyst.contrib.models.segmentation.unet import Unet

@registry.Model
class FinalSegment(Unet):
    pass


@registry.Model
class Linknet(Linknet):
    pass

@registry.Model
class ResNetUnetSimple(ResNetUnet):
    pass