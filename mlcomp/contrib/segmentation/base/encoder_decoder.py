import torch
import torch.nn as nn
import torch.nn.functional as F

from .model import Model


class EncoderDecoder(Model):
    def __init__(self, encoder, decoder, activation):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

        if callable(activation) or activation is None:
            self.activation = activation
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            raise ValueError(
                'Activation should be "sigmoid"/"softmax"/callable/None'
            )

    def forward(self, x):
        """Sequentially pass `x` trough model`s
         `encoder` and `decoder` (return logits!)"""
        input_shape = x.size()[-2:]

        x = self.encoder(x)
        x = self.decoder(x)

        if input_shape[0] != x.shape[-2] or input_shape[1] != x.shape[-1]:
            x = F.interpolate(
                x, size=input_shape, mode='bilinear', align_corners=False
            )
        return x

    def predict(self, x):
        """Inference method. Switch model to `eval` mode, call `.forward(x)`
        and apply activation function
         (if activation is not `None`) with `torch.no_grad()`

        Args:
            x: 4D torch tensor with shape (batch_size, channels, height, width)

        Return:
            prediction: 4D torch tensor with shape
            (batch_size, classes, height, width)

        """
        if self.training:
            self.eval()

        with torch.no_grad():
            x = self.forward(x)
            if self.activation:
                x = self.activation(x)

        return x
