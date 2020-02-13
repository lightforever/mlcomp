import timm
import torch.nn as nn


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class Timm(nn.Module):
    def __init__(self, variant, num_classes, pretrained=True, activation=None):
        super().__init__()

        model = timm.create_model(
            variant, pretrained=pretrained,
            num_classes=num_classes)

        self.model = model
        # self.model.fc = nn.Sequential(
        #     LambdaLayer(lambda x: x.unsqueeze_(0)),
        #     nn.AdaptiveAvgPool1d(self.model.fc.in_features),
        #     LambdaLayer(lambda x: x.squeeze_(0).view(x.size(0), -1)),
        #     self.model.fc
        # )

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
        res = self.model(x)
        if isinstance(res, tuple):
            res = res[0]
        if self.activation:
            res = self.activation(res)
        return res


__all__ = ['Timm']
