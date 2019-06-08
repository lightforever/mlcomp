import torch.nn as nn
import torch.nn.functional as F
from catalyst.contrib import registry
from catalyst.contrib.models.segmentation.unet import Unet

@registry.Model
class SimpleUnet(Unet):
    pass
