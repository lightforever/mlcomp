from timm.models import GenEfficientNet
from timm.models.gen_efficientnet import _decode_arch_def, _resolve_bn_args, \
    hard_swish, hard_sigmoid
from torchvision.models import MobileNetV2


class MobileNetV2Encoder(MobileNetV2):
    def __init__(self):
        super().__init__()
        del self.classifier

    def forward(self, x):
        for n in range(0, 2):
            x = self.features[n](x)
        x0 = x

        for n in range(2, 4):
            x = self.features[n](x)
        x1 = x

        for n in range(4, 7):
            x = self.features[n](x)
        x2 = x

        for n in range(7, 14):
            x = self.features[n](x)
        x3 = x

        for n in range(14, 19):
            x = self.features[n](x)

        x4 = x

        return [x4, x3, x2, x1, x0]

    def load_state_dict(self, state_dict, **kwargs):
        state_dict.pop('classifier.1.bias')
        state_dict.pop('classifier.1.weight')
        super().load_state_dict(state_dict, **kwargs)


class MobileNetV3Encoder(GenEfficientNet):
    def __init__(
        self, channel_multiplier=1.0, in_chans=3, **kwargs
    ):
        arch_def = [
            # stage 0, 112x112 in
            ['ds_r1_k3_s1_e1_c16_nre_noskip'],  # relu
            # stage 1, 112x112 in
            ['ir_r1_k3_s2_e4_c24_nre', 'ir_r1_k3_s1_e3_c24_nre'],  # relu
            # stage 2, 56x56 in
            ['ir_r3_k5_s2_e3_c40_se0.25_nre'],  # relu
            # stage 3, 28x28 in
            [
                'ir_r1_k3_s2_e6_c80', 'ir_r1_k3_s1_e2.5_c80',
                'ir_r2_k3_s1_e2.3_c80'
            ],
            # hard-swish
            # stage 4, 14x14in
            ['ir_r2_k3_s1_e6_c112_se0.25'],  # hard-swish
            # stage 5, 14x14in
            ['ir_r3_k5_s2_e6_c160_se0.25'],  # hard-swish
            # stage 6, 7x7 in
            ['cn_r1_k1_s1_c960'],  # hard-swish
        ]

        super().__init__(
            _decode_arch_def(arch_def),
            stem_size=16,
            channel_multiplier=channel_multiplier,
            bn_args=_resolve_bn_args(kwargs),
            act_fn=hard_swish,
            se_gate_fn=hard_sigmoid,
            se_reduce_mid=True,
            head_conv='efficient',
            in_chans=in_chans,
            **kwargs
        )

    def forward(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        x = self.act_fn(x, inplace=True)
        x0 = x

        for n in range(0, 2):
            x = self.blocks[n](x)
        x1 = x

        for n in range(2, 3):
            x = self.blocks[n](x)
        x2 = x

        for n in range(3, 5):
            x = self.blocks[n](x)
        x3 = x

        for n in range(5, 7):
            x = self.blocks[n](x)

        x4 = x

        return [x4, x3, x2, x1, x0]

url_map = {
    'mobilenetv2': 'https://download.pytorch.org/'
    'models/mobilenet_v2-b0353104.pth',
    'mobilenetv3': 'https://github.com/rwightman/'
    'pytorch-image-models/releases/download/'
    'v0.1-weights/mobilenetv3_100-35495452.pth'
}


def _get_pretrained_settings(encoder):
    pretrained_settings = {
        'imagenet': {
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'url': url_map[encoder],
            'input_space': 'RGB',
            'input_range': [0, 1]
        }
    }
    return pretrained_settings


mobilenet_encoders = {
    'mobilenetv2': {
        'encoder': MobileNetV2Encoder,
        'out_shapes': (1280, 96, 32, 24, 16),
        'pretrained_settings': _get_pretrained_settings('mobilenetv2'),
        'params': {}
    },
    'mobilenetv3': {
        'encoder': MobileNetV3Encoder,
        'out_shapes': (960, 112, 40, 24, 16),
        'pretrained_settings': _get_pretrained_settings('mobilenetv3'),
        'params': {}
    },
}
