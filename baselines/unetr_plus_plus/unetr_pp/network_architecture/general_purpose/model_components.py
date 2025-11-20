from torch import nn
from timm.models.layers import trunc_normal_
from typing import Sequence, Tuple, Union
from monai.networks.layers.utils import get_norm_layer
from monai.utils import optional_import
from unetr_pp.network_architecture.layers import LayerNorm
from unetr_pp.network_architecture.general_purpose.transformerblock import TransformerBlock
from unetr_pp.network_architecture.dynunet_block import get_conv_layer, UnetResBlock
import torch

einops, _ = optional_import("einops")


class UnetrPPEncoder(nn.Module):
    def __init__(self, input_size=None, dims=None,
                 proj_size=None, depths=None, num_heads=4, spatial_dims=3, in_channels=1, dropout=0.0,
                 transformer_dropout_rate=0.15, img_size=None,
                 **kwargs):
        super().__init__()

        if depths is None:
            depths = [3, 3, 3, 3]
        if proj_size is None:
            proj_size = [64, 64, 64, 32]
        if dims is None:
            dims = [32, 64, 128, 256]
        if input_size is None:
            input_size = [32 * 32 * 32, 16 * 16 * 16, 8 * 8 * 8, 4 * 4 * 4]

        print("Encoder input sizes:", input_size)
        print("Encoder dims:", dims)
        print("Encoder proj sizes:", proj_size)
        print("Encoder depths:", depths)

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem_layer = nn.Sequential(
            get_conv_layer(spatial_dims, in_channels, dims[0], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                           dropout=dropout, conv_only=True, ),
            get_norm_layer(name=("group", {"num_groups": in_channels}), channels=dims[0]),
        )
        self.downsample_layers.append(stem_layer)

        # calculate dimension of the input after stem layer
        print("calculating input size using img_size:", img_size)
        x = torch.rand(1, in_channels, img_size[0], img_size[1], img_size[2])
        x = self.downsample_layers[0](x)
        _, _, h, w, d = x.shape
        # round to the closest multiple of 2 (down)
        # h, w, d = h >> 1 << 1, w >> 1 << 1, d >> 1 << 1
        input_size[0] = h * w * d

        print("i got h,w,d:", h, w, d)
        print("Calculated input size after downsample:", input_size[0])

        # then the other inputs SHOULD be calculated based on that
        # for i in range(1, 4):
        #     input_size[i] = input_size[i - 1] // 8
        # DOESNT WORK; falling back to previous method of calculation

        for i in range(3):
            downsample_layer = nn.Sequential(
                get_conv_layer(spatial_dims, dims[i], dims[i + 1], kernel_size=(2, 2, 2), stride=(2, 2, 2),
                               dropout=dropout, conv_only=True, ),
                get_norm_layer(name=("group", {"num_groups": dims[i]}), channels=dims[i + 1]),
            )
            self.downsample_layers.append(downsample_layer)
            x = downsample_layer(x)
            _, _, h, w, d = x.shape
            # if i < 2:
            #     h, w, d = h >> 1 << 1, w >> 1 << 1, d >> 1 << 1
            input_size[i + 1] = h * w * d
            print(f"Calculated input size after downsample {i + 1}:", input_size[i + 1])

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple Transformer blocks
        for i in range(4):
            stage_blocks = []
            for j in range(depths[i]):
                stage_blocks.append(
                    TransformerBlock(input_size=input_size[i], hidden_size=dims[i], proj_size=proj_size[i],
                                     num_heads=num_heads,
                                     dropout_rate=transformer_dropout_rate, pos_embed=True))
            self.stages.append(nn.Sequential(*stage_blocks))
        self.hidden_states = []
        self.apply(self._init_weights)
        self.input_size = input_size

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (LayerNorm, nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        hidden_states = []
        print("forward_features input x shape:", x.shape)
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)

        hidden_states.append(x)

        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            if any([s % 2 != 0 for s in x.shape]) and i != 3:
                # fix by cropping if the size is odd
                x = x[
                    :,
                    :,
                    : x.shape[2] - (x.shape[2] % 2),
                    : x.shape[3] - (x.shape[3] % 2),
                    : x.shape[4] - (x.shape[4] % 2),
                ]
            x = self.stages[i](x)
            if i == 3:  # Reshape the output of the last stage
                x = einops.rearrange(x, "b c h w d -> b (h w d) c")
            hidden_states.append(x)
        return x, hidden_states

    def forward(self, x):
        x, hidden_states = self.forward_features(x)
        return x, hidden_states


class UnetrUpBlock(nn.Module):
    def __init__(
            self,
            spatial_dims: int,
            in_channels: int,
            out_channels: int,
            kernel_size: Union[Sequence[int], int],
            upsample_kernel_size: Union[Sequence[int], int],
            norm_name: Union[Tuple, str],
            proj_size: int = 64,
            num_heads: int = 4,
            out_size: int = 0,
            depth: int = 3,
            conv_decoder: bool = False,
    ) -> None:
        """
        Args:
            spatial_dims: number of spatial dimensions.
            in_channels: number of input channels.
            out_channels: number of output channels.
            kernel_size: convolution kernel size.
            upsample_kernel_size: convolution kernel size for transposed convolution layers.
            norm_name: feature normalization type and arguments.
            proj_size: projection size for keys and values in the spatial attention module.
            num_heads: number of heads inside each EPA module.
            out_size: spatial size for each decoder.
            depth: number of blocks for the current decoder stage.
        """

        super().__init__()
        upsample_stride = upsample_kernel_size
        self.transp_conv = get_conv_layer(
            spatial_dims,
            in_channels,
            out_channels,
            kernel_size=upsample_kernel_size,
            stride=upsample_stride,
            conv_only=True,
            is_transposed=True,
        )

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.decoder_block = nn.ModuleList()

        # If this is the last decoder, use ConvBlock(UnetResBlock) instead of EPA_Block (see suppl. material in the paper)
        if conv_decoder == True:
            self.decoder_block.append(
                UnetResBlock(spatial_dims, out_channels, out_channels, kernel_size=kernel_size, stride=1,
                             norm_name=norm_name, ))
        else:
            stage_blocks = []
            for j in range(depth):
                stage_blocks.append(TransformerBlock(input_size=out_size, hidden_size=out_channels, proj_size=proj_size,
                                                     num_heads=num_heads,
                                                     dropout_rate=0.15, pos_embed=True))
            self.decoder_block.append(nn.Sequential(*stage_blocks))

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, inp, skip):

        out = self.transp_conv(inp)
        print("UpBlock transp conv output shape:", out.shape)
        out = out + skip
        out = self.decoder_block[0](out)

        return out
