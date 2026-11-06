import torch
from torch import nn
from typing import Union, List, Tuple, Type
from torch.nn.modules.dropout import _DropoutNd
from dynamic_network_architectures.building_blocks.simple_conv_blocks import (
    StackedConvBlocks,
)
from dynamic_network_architectures.building_blocks.helper import get_matching_convtransp
from dynamic_network_architectures.building_blocks.residual_encoders import (
    ResidualEncoder,
)
from dynamic_network_architectures.building_blocks.plain_conv_encoder import (
    PlainConvEncoder,
)


class UNetDecoder(nn.Module):
    """Decode encoder skips into full-resolution foreground segmentation logits.

    Adapted from dynamic-network-architectures; encoder ownership and unused
    helpers were removed. See THIRD_PARTY_NOTICES.md for attribution.
    """

    def __init__(
        self,
        encoder: Union[PlainConvEncoder, ResidualEncoder],
        num_classes: int,
        n_conv_per_stage: Union[int, Tuple[int, ...], List[int]],
        deep_supervision,
        nonlin_first: bool = False,
        norm_op: Union[None, Type[nn.Module]] = None,
        norm_op_kwargs: dict = None,
        dropout_op: Union[None, Type[_DropoutNd]] = None,
        dropout_op_kwargs: dict = None,
        nonlin: Union[None, Type[torch.nn.Module]] = None,
        nonlin_kwargs: dict = None,
        conv_bias: bool = None,
    ):
        super().__init__()
        self.deep_supervision = deep_supervision
        self.num_classes = num_classes
        n_stages_encoder = len(encoder.output_channels)
        if isinstance(n_conv_per_stage, int):
            n_conv_per_stage = [n_conv_per_stage] * (n_stages_encoder - 1)
        assert len(n_conv_per_stage) == n_stages_encoder - 1, (
            "n_conv_per_stage must have as many entries as we have resolution stages - 1 (n_stages in encoder - 1), here: %d"
            % n_stages_encoder
        )
        transpconv_op = get_matching_convtransp(conv_op=encoder.conv_op)
        conv_bias = encoder.conv_bias if conv_bias is None else conv_bias
        norm_op = encoder.norm_op if norm_op is None else norm_op
        norm_op_kwargs = (
            encoder.norm_op_kwargs if norm_op_kwargs is None else norm_op_kwargs
        )
        dropout_op = encoder.dropout_op if dropout_op is None else dropout_op
        dropout_op_kwargs = (
            encoder.dropout_op_kwargs
            if dropout_op_kwargs is None
            else dropout_op_kwargs
        )
        nonlin = encoder.nonlin if nonlin is None else nonlin
        nonlin_kwargs = (
            encoder.nonlin_kwargs if nonlin_kwargs is None else nonlin_kwargs
        )
        stages = []
        transpconvs = []
        seg_layers = []
        for s in range(1, n_stages_encoder):
            input_features_below = encoder.output_channels[-s]
            input_features_skip = encoder.output_channels[-(s + 1)]
            stride_for_transpconv = encoder.strides[-s]
            transpconvs.append(
                transpconv_op(
                    input_features_below,
                    input_features_skip,
                    stride_for_transpconv,
                    stride_for_transpconv,
                    bias=conv_bias,
                )
            )
            stages.append(
                StackedConvBlocks(
                    n_conv_per_stage[s - 1],
                    encoder.conv_op,
                    2 * input_features_skip,
                    input_features_skip,
                    encoder.kernel_sizes[-(s + 1)],
                    1,
                    conv_bias,
                    norm_op,
                    norm_op_kwargs,
                    dropout_op,
                    dropout_op_kwargs,
                    nonlin,
                    nonlin_kwargs,
                    nonlin_first,
                )
            )
            if self.deep_supervision or s == n_stages_encoder - 1:
                seg_layers.append(
                    encoder.conv_op(
                        input_features_skip, num_classes, 1, 1, 0, bias=True
                    )
                )
        self.stages = nn.ModuleList(stages)
        self.transpconvs = nn.ModuleList(transpconvs)
        self.seg_layers = nn.ModuleList(seg_layers)

    def forward(self, skips):
        lres_input = skips[-1]
        seg_outputs = []
        for s in range(len(self.stages)):
            x = self.transpconvs[s](lres_input)
            x = torch.cat((x, skips[-(s + 2)]), 1)
            x = self.stages[s](x)
            if self.deep_supervision:
                seg_outputs.append(self.seg_layers[s](x))
            elif s == len(self.stages) - 1:
                seg_outputs.append(self.seg_layers[-1](x))
            lres_input = x
        seg_outputs = seg_outputs[::-1]
        if not self.deep_supervision:
            r = seg_outputs[0]
        else:
            r = seg_outputs
        return r
