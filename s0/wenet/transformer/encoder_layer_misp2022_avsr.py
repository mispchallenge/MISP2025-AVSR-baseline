# Copyright (c) 2023 USTC (Zhe Wang, Yusheng Dai)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Modified from ESPnet(https://github.com/espnet/espnet)

"""Encoder self-attention layer definition."""

from typing import Optional, Tuple

import torch
from torch import nn

class CrossAttentionFusionEncoderLayer(nn.Module):
    """Encoder layer module.

    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool): Whether to use layer_norm before the first block.
        concat_after (bool): Whether to concat attention layer's input and output.
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied. i.e. x -> x + att(x)
        stochastic_depth_rate (float): Proability to skip this layer.
            During training, the layer may skip residual computation and return input
            as-is with given probability.
    """

    def __init__(
        self,
        size,
        self_attn,
        src_attn,
        feed_forward,
        feed_forward_macaron,
        conv_module,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        src_first=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)  # for the MHA module
        self.norm_crossmha = nn.LayerNorm(size, eps=1e-5)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)  # for the CNN module
            self.norm_final = nn.LayerNorm(size, eps=1e-5)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        self.src_first = src_first
        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x_input, mask, video, video_mask,att_augmask=None,cache=None):
        """Compute encoded features.

        Args:
            x_input (Union[Tuple, torch.Tensor]): Input tensor w/ or w/o pos emb.
                - w/ pos emb: Tuple of tensors [(#batch, time, size), (1, time, size)].
                - w/o pos emb: Tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            att_augmask (torch.Tensor): Mask tensor for attention augmentation (#batch,time,time).
            cache (torch.Tensor): Cache tensor of the input (#batch, time - 1, size).
            

        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
            torch.Tensor: Encoded video (#batch, maxlen_in, size).
            torch.Tensor: Encoded video mask (#batch, maxlen_in).

        """
        if isinstance(x_input, tuple):
            x, pos_emb = x_input[0], x_input[1]
        else:
            x, pos_emb = x_input, None
        
        if isinstance(video, tuple):
            video = video[0]

        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)

        if skip_layer:
            if cache is not None:
                x = torch.cat([cache, x], dim=1)
            if pos_emb is not None:
                return (x, pos_emb), mask
            return x, mask

        #cross attention module 
        if self.src_first:
            residual = x
            if self.normalize_before:
                x = self.norm_crossmha(x) # note no normalization for video, so add norm to video before input to the ender layer
            if pos_emb is not None:
                x_att, _ = self.src_attn(torch.bmm(att_augmask, video) if att_augmask else video, x, x, mask, pos_emb)
                # x_att = self.src_attn(torch.bmm(att_augmask, video) if att_augmask else video, x, x, pos_emb, mask)
            else:
                x_att, _ = self.src_attn(torch.bmm(att_augmask, video) if att_augmask else video, x, x, mask)

            if self.concat_after:
                x_concat = torch.cat((x, x_att), dim=-1)
                x = residual + stoch_layer_coeff * self.concat_linear2(x_concat)
            else:
                x = residual + stoch_layer_coeff * self.dropout(x_att)
            if not self.normalize_before:
                x = self.norm_crossmha(x)

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
                self.feed_forward_macaron(x)
            )
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)

        if cache is None:
            x_q = x
        else:
            assert cache.shape == (x.shape[0], x.shape[1] - 1, self.size)
            x_q = x[:, -1:, :]
            residual = residual[:, -1:, :]
            mask = None if mask is None else mask[:, -1:, :]

        if pos_emb is not None:
            x_att, _ = self.self_attn(x_q, x, x, mask, pos_emb)
        else:
            x_att, _ = self.self_attn(x_q, x, x, mask)

        if self.concat_after:
            x_concat = torch.cat((x, x_att), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear1(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        #cross attention module 
        if not self.src_first:
            residual = x
            if self.normalize_before:
                x = self.norm_crossmha(x) # note no normalization for video, so add norm to video before input to the ender layer
            if pos_emb is not None:
                x_att, _ = self.src_attn(torch.bmm(att_augmask, video) if att_augmask else video, x, x, mask, pos_emb)
            else:
                x_att, _ = self.src_attn(torch.bmm(att_augmask, video) if att_augmask else video, x, x, mask)

            if self.concat_after:
                x_concat = torch.cat((x, x_att), dim=-1)
                x = residual + stoch_layer_coeff * self.concat_linear2(x_concat)
            else:
                x = residual + stoch_layer_coeff * self.dropout(x_att)
            if not self.normalize_before:
                x = self.norm_crossmha(x)

        # convolution module
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            # print(type(x))
            # print(self.conv_module(x))
            # x = residual + stoch_layer_coeff * self.dropout(self.conv_module(x))
            x, _ = self.conv_module(x)
            x = residual + stoch_layer_coeff * self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + stoch_layer_coeff * self.ff_scale * self.dropout(
            self.feed_forward(x)
        )
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        if cache is not None:
            x = torch.cat([cache, x], dim=1)

        if pos_emb is not None:
            return (x, pos_emb),mask,video,video_mask
        return x,mask,video,video_mask