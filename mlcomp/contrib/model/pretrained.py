import torch.nn as nn
import torch

import pretrainedmodels


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Pretrained(nn.Module):
    def __init__(self, variant, num_classes, pretrained=True, activation=None):
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
            self.last = nn.Linear(model.last_linear.in_features, num_classes)
        else:
            self.model = model
            linear = self.model.last_linear
            if isinstance(linear, nn.Linear):
                self.model.last_linear = nn.Linear(
                    model.last_linear.in_features,
                    num_classes
                )
                self.model.last_linear.in_channels = linear.in_features
            elif isinstance(linear, nn.Conv2d):
                self.model.last_linear = nn.Conv2d(
                    linear.in_channels,
                    num_classes,
                    kernel_size=linear.kernel_size,
                    bias=True
                )
                self.model.last_linear.in_features = linear.in_channels

        self.model.last_linear = nn.Sequential(
            LambdaLayer(lambda x: x.unsqueeze_(0)),
            nn.AdaptiveAvgPool1d(self.model.last_linear.in_channels),
            LambdaLayer(lambda x: x.squeeze_(0).view(x.size(0), -1)),
            self.model.last_linear
        )

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                'Activation should be "sigmoid"/"softmax"/callable/None')

    def forward(self, x):
        if not self.need_refactor:
            res = self.model(x)
            if isinstance(res, tuple):
                return res[0]
            return res
        x = self.l1(x)
        x = x.view(x.size()[0], -1)
        x = self.last(x)
        if self.activation:
            x = self.activation(x)
        return x


__all__ = ['Pretrained']
