#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2022 USTC Dalison
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Encoder self-attention layer definition."""
from typing import Optional
from typing import Tuple
import torch
import logging 
from torch import nn
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
import torch
from typeguard import check_argument_types
from espnet.nets.pytorch_backend.transformer.embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder_layer import EncoderLayer
from espnet.nets.pytorch_backend.transformer.layer_norm import LayerNorm
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import Conv1dLinear
from espnet.nets.pytorch_backend.transformer.multi_layer_conv import MultiLayeredConv1d
from espnet.nets.pytorch_backend.transformer.positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from espnet.nets.pytorch_backend.transformer.repeat import repeat
from espnet.nets.pytorch_backend.transformer.subsampling import check_short_utt
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling2
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling6
from espnet.nets.pytorch_backend.transformer.subsampling import Conv2dSubsampling8
from espnet.nets.pytorch_backend.transformer.subsampling import TooShortUttError
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.encoder.utils import DimConvert
from espnet.nets.pytorch_backend.transformer.attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
    LegacyRelPositionMultiHeadedAttention,  # noqa: H301
    TraditionMultiheadRelativeAttention,
)

class ChannelwiseLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
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
        feed_forward,
        dropout_rate,
        normalize_before=True,
        concat_after=False,
        stochastic_depth_rate=0.0,
    ):
        """Construct an ChannelwiseLayer object."""
        super(ChannelwiseLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size)
        self.norm2 = LayerNorm(size)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        self.stochastic_depth_rate = stochastic_depth_rate

    def forward(self, x, mask):
        """Compute encoded features.

        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).

        """
        skip_layer = False
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.
        stoch_layer_coeff = 1.0
        if self.training and self.stochastic_depth_rate > 0:
            skip_layer = torch.rand(1).item() < self.stochastic_depth_rate
            stoch_layer_coeff = 1.0 / (1 - self.stochastic_depth_rate)
        #MHA 
        residual = x
        x = self.norm1(x)
        x_q = x
        if self.concat_after:
            x_concat = torch.cat((x, self.self_attn(x_q, x, x, mask)), dim=-1)
            x = residual + stoch_layer_coeff * self.concat_linear(x_concat)
        else:
            x = residual + stoch_layer_coeff * self.dropout(
                self.self_attn(x_q, x, x, mask)
            )
        #forward layer
        residual = x
        x = self.norm2(x)
        x = residual + stoch_layer_coeff * self.dropout(self.feed_forward(x))

        return x, mask

class AttentionKQVFusion(nn.Module):
    """CrosschannelLayer layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention` instance
            can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward`, `MultiLayeredConv1d`, or `Conv1dLinear` instance
            can be used as the argument.
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
        src_attn,
        feed_forward,
        dropout_rate,
        normalize_before=True,
        input_mask:bool=True,
        output_mask:bool=True,
    ):
        """Construct an channelwiseLayer object."""
        super(AttentionKQVFusion, self).__init__()
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = LayerNorm(size) #for k
        self.norm2 = LayerNorm(size) #for q,v
        self.norm3 = LayerNorm(size) 
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.input_mask=input_mask
        self.output_mask=output_mask

    def forward(self, x_q, x_kv, mask):
        """Compute encoded features.
        Args:
            x_input (torch.Tensor): Input tensor (#batch, time, size).
            mask (torch.Tensor): Mask tensor for the input (#batch, time).
            Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time).
        """

        if not self.input_mask:
            mask = (~make_pad_mask(mask)[:, None, :]).to(x_q.device) 
        # with stochastic depth, residual connection `x + f(x)` becomes
        # `x <- x + 1 / (1 - p) * f(x)` at training time.

        #pos emb
        pos_emb = None
        if isinstance(x_kv, tuple):
            x_kv, pos_emb = x_kv[0], x_kv[1]
        if isinstance(x_q, tuple):
            x_q = x_q[0]
            
        #MHA 
        residual = x_kv
        if self.normalize_before:
            x_q = self.norm1(x_q)
            x_kv = self.norm2(x_kv)
        if pos_emb != None :
            x = residual + self.dropout(
                    self.src_attn(x_q, x_kv, x_kv, pos_emb, mask)
                )
        else:
            x = residual + self.dropout(
                    self.src_attn(x_q, x_kv, x_kv, mask)
                )
        if not self.normalize_before:
            x = self.norm1(x)
    
        #forward layer
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)
        
        if not self.output_mask:
            mask = mask.squeeze(1).sum(1)
        
        if pos_emb is not None:
            return (x, pos_emb), mask
            
        return x, mask

class AVchannelwiseEncoderlayer(nn.Module):
    def __init__(
    self,
    channel_num,
    size,
    self_attn,
    feed_forward,
    dropout_rate,
    normalize_before=True,
    concat_after=False,
    stochastic_depth_rate=0.0,
    ):  
        super(AVchannelwiseEncoderlayer,self).__init__()
        self.avfusionlayer = ChannelwiseLayer(size,self_attn,feed_forward,dropout_rate,normalize_before,concat_after,stochastic_depth_rate)
        self.channelwiselayers = nn.ModuleList([ChannelwiseLayer(size,self_attn,feed_forward,dropout_rate,normalize_before,concat_after,stochastic_depth_rate) for _ in range(channel_num)])
        self.AttentionKQVFusiones = nn.ModuleList([AttentionKQVFusion(size,self_attn,feed_forward,dropout_rate,normalize_before,concat_after,stochastic_depth_rate) for _ in range(channel_num)])
        self.dimconver = DimConvert(in_channels=size*2,out_channels=size)
    def forward(self,mask,channels,av_embs,cross_channel_embs=None):
        #except the first layer because there isn't cross_channel_embed
        if cross_channel_embs != None:
            av_concat = torch.cat((av_embs, cross_channel_embs), dim=-1)
            av_embs = self.dimconver(av_concat) #B,T,D
        a_embs_res = [self.channelwiselayers[i](channel,mask) for i,channel in enumerate(channels)] #
        av_embs_res = self.avfusionlayer(av_embs,mask) # av_embs after cross-channel Branch,masks
        fusion_embs_res = [self.AttentionKQVFusiones[channel_id](x_q = a_emb,x_kv = av_embs_res[0], mask=mask) for channel_id, (a_emb, mask) in enumerate(a_embs_res)]
        output_channels = [item[0] for item in fusion_embs_res] #[,] each item in the list is shaped  B,T,D
        cross_channel_embs = torch.stack(output_channels, dim = 0).mean(dim=0) #B,T,D
        return fusion_embs_res[0][1],output_channels,av_embs_res[0],cross_channel_embs # mask,channel wise outputs embedding,av embedding and cross_channel_embs

class AVchannelwiseEncoder(AbsEncoder):
    """AVchannelwiseEncoder encoder module.

    Args:
        input_size: input dim
        output_size: dimension of attention
        attention_heads: the number of heads of multi head attention
        linear_units: the number of units of position-wise feed forward
        num_blocks: the number of decoder blocks
        dropout_rate: dropout rate
        attention_dropout_rate: dropout rate in attention
        positional_dropout_rate: dropout rate after adding positional encoding
        input_layer: input layer type
        pos_enc_class: PositionalEncoding or ScaledPositionalEncoding
        normalize_before: whether to use layer_norm before the first block
        concat_after: whether to concat attention layer's input and output
            if True, additional linear will be applied.
            i.e. x -> x + linear(concat(x, att(x)))
            if False, no additional linear will be applied.
            i.e. x -> x + att(x)
        positionwise_layer_type: linear of conv1d
        positionwise_conv_kernel_size: kernel size of positionwise conv1d layer
        padding_idx: padding_idx for input_layer=embed
    """

    def __init__(
        self,
        a_size: int,
        v_size:int,
        channel_num: int = 6,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        pos_enc_class=PositionalEncoding,
        normalize_before: bool = True,
        concat_after: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 1,
        padding_idx: int = -1,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.channel_num = channel_num
        self.a_size = a_size
        self.channel_embed = Conv2dSubsampling(self.a_size, output_size, dropout_rate)
        self.clean_embed = Conv2dSubsampling(self.a_size, output_size, dropout_rate)
        self.fusion_layer = DimConvert(in_channels=output_size+v_size,out_channels=output_size)
        
        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                output_size,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        
        self.encoder = repeat(
            num_blocks,
            lambda lnum: AVchannelwiseEncoderlayer(
                self.channel_num,
                output_size,
                MultiHeadedAttention(
                    attention_heads, output_size, attention_dropout_rate
                ),
                positionwise_layer(*positionwise_layer_args),
                dropout_rate,
                normalize_before,
                concat_after,
            ),
        )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        a_frames: torch.Tensor,
        a_lengths: torch.Tensor, 
        v_frames: torch.Tensor,
        v_lengths: torch.Tensor,
        channels: torch.Tensor, # (CxB)XTXD
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Embed positions in tensor.

        Args:
            xs_pad: input tensor (B, L, D)
            ilens: input length (B)
            prev_states: Not to be used now.
        Returns:
            position embedded tensor and mask
        """
        #mask
        masks = (~make_pad_mask(a_lengths)[:, None, :]).to(a_frames.device) #B,1,T
        batch_size = a_frames.shape[0]
        c_masks = masks.repeat(self.channel_num ,1,1)
        short_status, limit_size = check_short_utt(self.clean_embed, a_frames.size(1))
        if short_status:
            raise TooShortUttError(
                f"has {a_frames.size(1)} frames and is too short for subsampling "
                + f"(it needs more than {limit_size} frames), return empty results",
                a_frames.size(1),
                limit_size,
            )
        #2dconv downsumpling for channels and a_frams
        channels = self.channel_embed(channels, c_masks)[0]  # (channelsxB)xTxD
        channels = channels.reshape(self.channel_num,batch_size,-1,self._output_size) # (channelsxB)xTxD -> channelsxBxTxD
        a_frames, masks = self.clean_embed(a_frames, masks)
        #aligment
        a_lengths = ((a_lengths - 1) // 2 - 1) // 2
        a_lengths = a_lengths.min(v_lengths)
        channels = [channel[:,:max(a_lengths)] for channel in channels]
        a_frames = a_frames[:,:max(a_lengths)]
        v_frames = v_frames[:,:max(a_lengths)]
        v_lengths = a_lengths
        #av fusion 
        av_frames = self.fusion_layer(torch.cat((a_frames, v_frames), dim=-1))   
        #encoder 12 block 
        masks,output_channels,av_embs,cross_channel_embs = self.encoder(masks,channels,av_frames,None)
        #post processing
        if self.normalize_before:
            xs_pad = self.after_norm(cross_channel_embs)
        olens = masks.squeeze(1).sum(1)
        return xs_pad, olens, None

if __name__ == "__main__": 
    #test AttentionKQVFusion
    def get_posemb_attention(MHA_type,MHA_conf,size):
        if MHA_type == "abs_pos":
            embed_layer = PositionalEncoding(size,MHA_conf["dropout_rate"])
            attetion_layer = MultiHeadedAttention(**MHA_conf,n_feat=size)
        elif MHA_type == "rel_pos":
            embed_layer = RelPositionalEncoding(size,MHA_conf["dropout_rate"])
            attetion_layer = RelPositionMultiHeadedAttention(**MHA_conf,n_feat=size)
        elif MHA_type == "trad_rel_pos":
            embed_layer = None
            attetion_layer = TraditionMultiheadRelativeAttention(
                num_heads=MHA_conf["n_head"],
                embed_dim=size,
                dropout=MHA_conf["dropout_rate"]
                )
        else: 
            logging.error(f"MHA_type must in abs_pos,rel_pos,trad_rel_pos,but got {MHA_type}")
        return embed_layer,attetion_layer
    
    def get_positionwise_layer(
        positionwise_layer_type="linear",
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        positionwise_conv_kernel_size=1,
    ):
        """Define positionwise layer."""
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (attention_dim, linear_units, dropout_rate)
        elif positionwise_layer_type == "conv1d":
            positionwise_layer = MultiLayeredConv1d
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        elif positionwise_layer_type == "conv1d-linear":
            positionwise_layer = Conv1dLinear
            positionwise_layer_args = (
                attention_dim,
                linear_units,
                positionwise_conv_kernel_size,
                dropout_rate,
            )
        else:
            raise NotImplementedError("Support only linear or conv1d.")
        return positionwise_layer, positionwise_layer_args
    yaml_path = "/yrfs2/cv1/hangchen2/espnet/mispi/avsr/conf/avsrfinetune/avsr_com_finetune.yaml"
    import yaml 
    with open(yaml_path) as f :
        cfg = yaml.safe_load(f)
    conformer_conf = cfg["encoder_conf"]
    attentionfusion_conf = cfg["encoder_conf"]["attentionfusion_conf"]

    positionwise_layer, positionwise_layer_args = get_positionwise_layer(
         **attentionfusion_conf["positionwise_layer_args"],attention_dim=conformer_conf["output_size"])

    fusion_embedlayer, fusion_attentionlayer = get_posemb_attention(attentionfusion_conf["MHA_type"],attentionfusion_conf["MHA_conf"],conformer_conf["output_size"] )
    fusionblock = AttentionKQVFusion(size=conformer_conf["output_size"],
                                                embed_layer=fusion_embedlayer,
                                                src_attn=fusion_attentionlayer,
                                                feed_forward=positionwise_layer(*positionwise_layer_args),
                                                dropout_rate=attentionfusion_conf["dropout_rate"],
                                                normalize_before=attentionfusion_conf["normalize_before"],
                                                input_mask=False,           
                                                output_mask=False)
    lens=torch.randint(128,129,(32,))                                  
    x_q = torch.rand(32,128,512)
    x_kv = torch.rand(32,128,512)
    output,olen = fusionblock(x_q, x_kv, lens)
    print(output.shape,olen)
    #test AttentionKQVFusion end 

    # encoder = AVchannelwiseEncoder(a_size = 80,v_size = 512, channel_num = 6, output_size = 256,linear_units = 2048,dropout_rate = 0.1,)
    # channels = torch.rand(32*6,100,80) 
    # a_frames = torch.rand(32,100,80)
    # v_frames = torch.rand(32,100,512)
    # a_lengths = torch.randint(90,100,(32,))
    # v_lengths = torch.randint(90,100,(32,))
    # xs_pad, olens,_ = encoder(channels=channels,a_frames=a_frames,v_frames=v_frames,a_lengths=a_lengths,v_lengths=v_lengths)
    # print(xs_pad.shape,olens.shape)

