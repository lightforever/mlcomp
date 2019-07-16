import torch.nn as nn
import torch

import pretrainedmodels


class Pretrained(nn.Module):
    def __init__(self, variant, n_classes, pretrained=True):
        super().__init__()
        params = {'num_classes': 1000}
        if not pretrained:
            params['pretrained'] = None
        model = pretrainedmodels.__dict__[variant](**params)
        self.need_refactor = False
        if 'resnet' in variant:
            self.need_refactor = True

        if self.need_refactor:
            self.l1 = nn.Sequential(*list(model.children())[:-1])
            if torch.cuda.is_available():
                self.l1 = self.l1.to('cuda:0')
            self.last = nn.Linear(model.last_linear.in_features, n_classes)
        else:
            self.model = model
            linear = self.model.last_linear
            if isinstance(linear, nn.Linear):
                self.model.last_linear = nn.Linear(
                    model.last_linear.in_features,
                    n_classes
                )
            elif isinstance(linear, nn.Conv2d):
                self.model.last_linear = nn.Conv2d(
                    linear.in_channels,
                    n_classes,
                    kernel_size=linear.kernel_size,
                    bias=True
                )

    def forward(self, x):
        if not self.need_refactor:
            res = self.model(x)
            if isinstance(res, tuple):
                return res[0]
            return res
        x = self.l1(x)
        x = x.view(x.size()[0], -1)
        x = self.last(x)
        return x


__all__ = ['Pretrained']
