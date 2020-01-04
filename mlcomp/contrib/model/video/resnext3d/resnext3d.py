import torch
import torch.nn as nn

from .resnext3d_stage import ResStage
from .resnext3d_stem import R2Plus1DStem, ResNeXt3DStem

model_stems = {
    "r2plus1d_stem": R2Plus1DStem,
    "resnext3d_stem": ResNeXt3DStem,
    # For more types of model stem, add them below
}


class FullyConvolutionalLinear(nn.Module):
    def __init__(self, dim_in, num_classes):
        super(FullyConvolutionalLinear, self).__init__()
        # Perform FC in a fully convolutional manner. The FC layer will be
        # initialized with a different std comparing to convolutional layers.
        self.projection = nn.Linear(dim_in, num_classes, bias=True)

    def forward(self, x):
        # (N, C, T, H, W) -> (N, T, H, W, C).
        x = x.permute((0, 2, 3, 4, 1))
        x = self.projection(x)
        x = x.view(x.shape[0], -1)
        return x


class ResNeXt3D(torch.nn.Module):
    """
    Implementation of:
        1. Conventional `post-activated 3D ResNe(X)t <https://arxiv.org/
        abs/1812.03982>`_.

        2. `Pre-activated 3D ResNe(X)t <https://arxiv.org/abs/1811.12814>`_.
        The model consists of one stem, a number of stages, and one or multiple
        heads that are attached to different blocks in the stage.
    """

    def __init__(
            self,
            input_planes: int = 3,
            skip_transformation_type: str = 'postactivated_shortcut',
            residual_transformation_type: str = 'basic_transformation',
            num_blocks: list = (2, 2, 2, 2),
            stem_name: str = 'resnext3d_stem',
            stem_planes: int = 64,
            stem_temporal_kernel: int = 3,
            stem_spatial_kernel: int = 7,
            stem_maxpool: bool = False,
            stage_planes: int = 64,
            stage_temporal_kernel_basis: list = ([3], [3], [3], [3]),
            temporal_conv_1x1: list = (False, False, False, False),
            stage_temporal_stride: list = (1, 2, 2, 2),
            stage_spatial_stride: list = (1, 2, 2, 2),
            num_groups: int = 1,
            width_per_group: int = 64,
            zero_init_residual_transform: bool = False,
            pool_size: list = (1, 7, 7),
            in_plane: int = 512,
            num_classes: int = 2
    ):
        """
        Args:
            input_planes (int): the channel dimension of the input.
                Normally 3 is used for rgb input.
            skip_transformation_type (str): the type of skip transformation.
                residual_transformation_type (str):
                the type of residual transformation.
            num_blocks (list): list of the number of blocks in stages.
            stem_name (str): name of model stem.
            stem_planes (int): the output dimension
                of the convolution in the model stem.
            stem_temporal_kernel (int): the temporal kernel
                size of the convolution
                in the model stem.
            stem_spatial_kernel (int): the spatial kernel size
                of the convolution in the model stem.
            stem_maxpool (bool): If true, perform max pooling.
            stage_planes (int): the output channel dimension
                of the 1st residual stage
            stage_temporal_kernel_basis (list): Basis of temporal kernel
                sizes for each of the stage.
            temporal_conv_1x1 (bool): Only useful for BottleneckTransformation.
                In a pathaway, if True, do temporal convolution
                in the first 1x1
                Conv3d. Otherwise, do it in the second 3x3 Conv3d.
            stage_temporal_stride (int): the temporal stride of the residual
                transformation.
            stage_spatial_stride (int): the spatial stride of the the residual
                transformation.
            num_groups (int): number of groups for the convolution.
                num_groups = 1 is for standard ResNet like networks, and
                num_groups > 1 is for ResNeXt like networks.
            width_per_group (int): Number of channels per group in 2nd (group)
                conv in the residual transformation in the first stage
            zero_init_residual_transform (bool): if true, the weight of last
                operation, which could be either BatchNorm3D in post-activated
                transformation or Conv3D in pre-activated transformation,
                in the residual transformation is initialized to zero
            pool_size: for fully convolution layer
            in_plane: for fully convolution layer
            num_classes: number of classes
        """

        super().__init__()

        num_stages = len(num_blocks)
        out_planes = [stage_planes * 2 ** i for i in range(num_stages)]
        in_planes = [stem_planes] + out_planes[:-1]
        inner_planes = [
            num_groups * width_per_group * 2 ** i for i in range(num_stages)
        ]

        self.stem = model_stems[stem_name](
            stem_temporal_kernel,
            stem_spatial_kernel,
            input_planes,
            stem_planes,
            stem_maxpool,
        )

        stages = []
        for s in range(num_stages):
            stage = ResStage(
                s + 1,
                # stem is viewed as stage 0, and following stages start from 1
                [in_planes[s]],
                [out_planes[s]],
                [inner_planes[s]],
                [stage_temporal_kernel_basis[s]],
                [temporal_conv_1x1[s]],
                [stage_temporal_stride[s]],
                [stage_spatial_stride[s]],
                [num_blocks[s]],
                [num_groups],
                skip_transformation_type,
                residual_transformation_type,
                disable_pre_activation=(s == 0),
                final_stage=(s == (num_stages - 1)),
            )
            stages.append(stage)

        self.stages = nn.Sequential(*stages)
        self._init_parameter(zero_init_residual_transform)

        self.final_avgpool = nn.AvgPool3d(pool_size, stride=1)
        self.head_fcl = FullyConvolutionalLinear(
            in_plane, num_classes
        )

    def _init_parameter(self, zero_init_residual_transform):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                if (
                        hasattr(m, "final_transform_op")
                        and m.final_transform_op
                        and zero_init_residual_transform
                ):
                    nn.init.constant_(m.weight, 0)
                else:
                    nn.init.kaiming_normal_(
                        m.weight, mode="fan_out", nonlinearity="relu"
                    )
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d) and m.affine:
                if (
                        hasattr(m, "final_transform_op")
                        and m.final_transform_op
                        and zero_init_residual_transform
                ):
                    batchnorm_weight = 0.0
                else:
                    batchnorm_weight = 1.0
                nn.init.constant_(m.weight, batchnorm_weight)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: Tensor(B, C, T, W, H)
        """
        x = x.permute((0, 4, 1, 2, 3))

        out = self.stem([x])
        out = self.stages(out)[0]
        out = self.final_avgpool(out)
        out = self.head_fcl(out)

        return out
