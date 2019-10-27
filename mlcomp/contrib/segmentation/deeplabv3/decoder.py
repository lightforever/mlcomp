import torch
from mlcomp.contrib.segmentation.unet.decoder import DecoderBlock

from mlcomp.contrib.segmentation.base.model import Model
from torch import nn
from torch.nn import functional as F


class DeepLabHead(nn.Sequential):
    def __init__(self, num_classes):
        super(DeepLabHead, self).__init__(
            nn.Conv2d(256, 256, 3, padding=1, bias=False), nn.BatchNorm2d(256),
            nn.ReLU(), nn.Conv2d(256, num_classes, 1)
        )


class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(
                in_channels,
                out_channels,
                3,
                padding=dilation,
                dilation=dilation,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ]
        super(ASPPConv, self).__init__(*modules)


class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU()
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(
            x, size=size, mode='bilinear', align_corners=False
        )


class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates):
        super(ASPP, self).__init__()
        out_channels = 256
        modules = []
        modules.append(
            nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.BatchNorm2d(out_channels), nn.ReLU()
            )
        )

        rate1, rate2, rate3 = tuple(atrous_rates)
        modules.append(ASPPConv(in_channels, out_channels, rate1))
        modules.append(ASPPConv(in_channels, out_channels, rate2))
        modules.append(ASPPConv(in_channels, out_channels, rate3))
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels), nn.ReLU(), nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)


class DeepLabV3Decoder(Model):
    def __init__(
        self,
        encoder_channels,
        final_channels=1,
        use_batchnorm=True,
        attention_type=None,
        decoder_channels=(2048, 1024, 512),
    ):
        super().__init__()
        self.aspp = ASPP(
            in_channels=decoder_channels[-1], atrous_rates=[12, 24, 36]
        )
        self.head = DeepLabHead(final_channels)

        in_channels = self.compute_channels(encoder_channels, decoder_channels)
        out_channels = decoder_channels

        self.layer1 = DecoderBlock(
            in_channels[0],
            out_channels[0],
            use_batchnorm=use_batchnorm,
            attention_type=attention_type
        )
        self.layer2 = DecoderBlock(
            in_channels[1],
            out_channels[1],
            use_batchnorm=use_batchnorm,
            attention_type=attention_type
        )
        self.layer3 = DecoderBlock(
            in_channels[2],
            out_channels[2],
            use_batchnorm=use_batchnorm,
            attention_type=attention_type
        )
        self.initialize()

    def forward(self, x):
        encoder_head = x[0]
        skips = x[1:]

        x = self.layer1([encoder_head, skips[0]])
        x = self.layer2([x, skips[1]])
        x = self.layer3([x, skips[2]])

        x = self.aspp(x)
        x = self.head(x)
        return x

    def compute_channels(self, encoder_channels, decoder_channels):
        channels = [
            encoder_channels[0] + encoder_channels[1],
            encoder_channels[2] + decoder_channels[0],
            encoder_channels[3] + decoder_channels[1],
            encoder_channels[4] + decoder_channels[2]
        ]
        return channels
