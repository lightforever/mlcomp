from .decoder import DeepLabV3Decoder
from ..base import EncoderDecoder
from ..encoders import get_encoder


class DeepLabV3(EncoderDecoder):
    """
    Implements DeepLabV3 model from
    `"Rethinking Atrous Convolution for Semantic Image Segmentation"
    <https://arxiv.org/abs/1706.05587>`_.
    Args:
        encoder_name: name of classification model
        (without last dense layers) used as feature
            extractor to build segmentation model.
        encoder_weights: one of ``None`` (random initialization),
        ``imagenet`` (pre-training on ImageNet).
        classes: a number of classes for output
         (output shape - ``(batch, classes, h, w)``).
        activation: activation function used
        in ``.predict(x)`` method for inference.
            One of [``sigmoid``, ``softmax``, callable, None]
    Returns:
        ``torch.nn.Module``: **DeepLabV3**
    """

    def __init__(
        self,
        encoder_name='resnet34',
        encoder_weights='imagenet',
        classes=1,
        activation='sigmoid'
    ):
        encoder = get_encoder(encoder_name, encoder_weights=encoder_weights)

        decoder = DeepLabV3Decoder(
            encoder_channels=encoder.out_shapes,
            final_channels=classes,
        )

        super().__init__(encoder, decoder, activation)

        self.name = 'u-{}'.format(encoder_name)
