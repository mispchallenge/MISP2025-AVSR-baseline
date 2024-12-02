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

"""Encoder definition."""
from typing import Tuple

import torch
from typeguard import check_argument_types

from wenet.transformer.attention import MultiHeadedAttention
from wenet.transformer.attention import RelPositionMultiHeadedAttention
from wenet.transformer.convolution import ConvolutionModule
from wenet.transformer.embedding import PositionalEncoding
from wenet.transformer.embedding import RelPositionalEncoding
from wenet.transformer.embedding import NoPositionalEncoding
from wenet.transformer.encoder_layer import ConformerEncoderLayer
from wenet.transformer.positionwise_feed_forward import PositionwiseFeedForward
from wenet.transformer.subsampling import NoSubsampling
from wenet.transformer.subsampling import Conv2dSubsampling4
from wenet.transformer.subsampling import Conv2dSubsampling6
from wenet.transformer.subsampling import Conv2dSubsampling8
from wenet.transformer.subsampling import LinearNoSubsampling
from wenet.utils.common import get_activation
from wenet.utils.mask import make_pad_mask
from wenet.utils.mask import add_optional_chunk_mask

import logging
from wenet.transformer.encoder_layer_misp2022_avsr import CrossAttentionFusionEncoderLayer
from espnet2.asr.encoder.utils import NewDimConvert
"""
NewAVCrossAttentionEncoder is used for feat vggpreencode 25ps +video 25 ps ,different with other encoder it use audio feat as K, and speech_feat as K,Q, and it will be fine-tune based on a 12-layer asr comferformer
"""
class NewAVCrossAttentionEncoder(torch.nn.Module): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        input_size_a: int,
        input_size_v: int,
        v_num_blocks: int = 3,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        crossfusion_num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer_a: str = "conv2d",
        input_layer_v: str = "",
        normalize_before: bool = True,
        concat_after: bool = False,
        src_first: bool = False,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        srcattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        global_cmvn: torch.nn.Module = None,
        cnn_module_norm: str = "batch_norm",
    ):  
        assert check_argument_types()
        assert v_num_blocks!=0
        self.v_num_blocks = v_num_blocks
        self.crossfusion_num_blocks = crossfusion_num_blocks
        super().__init__()
        self._output_size = output_size
        if src_first:
            logging.info("using src_first")
        if rel_pos_type == "legacy":
            if pos_enc_layer_type == "rel_pos":
                pos_enc_layer_type = "legacy_rel_pos"
            if selfattention_layer_type == "rel_selfattn":
                selfattention_layer_type = "legacy_rel_selfattn"
        elif rel_pos_type == "latest":
            assert selfattention_layer_type != "legacy_rel_selfattn"
            assert pos_enc_layer_type != "legacy_rel_pos"
        else:
            raise ValueError("unknown rel_pos_type: " + rel_pos_type)

        activation = get_activation(activation_type)
        pos_enc_class = NewAVCrossAttentionEncoder.get_posembclass(pos_enc_layer_type,selfattention_layer_type)

        # Audio Embedding+CMVN
        if input_layer_a == "linear":
            subsampling_class_a = LinearNoSubsampling
        elif input_layer_a == "conv2d":
            subsampling_class_a = Conv2dSubsampling4
        elif input_layer_a == "conv2d6":
            subsampling_class_a = Conv2dSubsampling6
        elif input_layer_a == "conv2d8":
            subsampling_class_a = Conv2dSubsampling8
        elif input_layer_a == "":
            # Add: zhewang18
            subsampling_class_a = NoSubsampling
        else:
            raise ValueError("unknown input_layer_a: " + input_layer_a)

        self.global_cmvn = global_cmvn
        
        self.embed_a = subsampling_class_a(
            input_size_a,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
        )
        # Video Embedding
        if input_layer_v == "linear":
            subsampling_class_v = LinearNoSubsampling
        elif input_layer_v == "conv2d":
            subsampling_class_v = Conv2dSubsampling4
        elif input_layer_v == "conv2d6":
            subsampling_class_v = Conv2dSubsampling6
        elif input_layer_v == "conv2d8":
            subsampling_class_v = Conv2dSubsampling8
        elif input_layer_v == "":
            # Add: zhewang18
            subsampling_class_v = NoSubsampling
        else:
            raise ValueError("unknown input_layer_a: " + input_layer_v)
        
        self.embed_v = subsampling_class_v(
            input_size_v,
            output_size,
            dropout_rate,
            pos_enc_class(output_size, positional_dropout_rate),
        )
        
        vencoder_selfattn_layer,vencoder_selfattn_layer_args = NewAVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    attention_dropout_rate)
        encoder_selfattn_layer,encoder_selfattn_layer_args = NewAVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    attention_dropout_rate)
        encoder_srcattn_layer,encoder_srcattn_layer_args = NewAVCrossAttentionEncoder.getattentionMHA(srcattention_layer_type,
                                                                                pos_enc_layer_type,
                                                                                attention_heads,
                                                                                output_size,
                                                                                attention_dropout_rate)

        self.normalize_before = normalize_before
        positionwise_layer,positionwise_layer_args = NewAVCrossAttentionEncoder.get_positionwise_layer(output_size,
                                                                                                       linear_units,
                                                                                                       dropout_rate,
                                                                                                       activation)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm)
        
        self.v_encoderlayers = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                vencoder_selfattn_layer(*vencoder_selfattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
            ) for _ in range(v_num_blocks)
        ])  

        self.vmemory_fusion = NewDimConvert(in_channels=output_size*v_num_blocks,out_channels=output_size)  
        self.vmemory_fusion_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        

        self.cross_fusion_encoderlayers = torch.nn.ModuleList([
            CrossAttentionFusionEncoderLayer(
                output_size,
                encoder_selfattn_layer(*encoder_selfattn_layer_args),
                encoder_srcattn_layer(*encoder_srcattn_layer_args),
                positionwise_layer(*positionwise_layer_args),
                positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
                concat_after,
                src_first,
            ) for _ in range(crossfusion_num_blocks)
        ])
        
        if self.normalize_before:
            self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)

    @staticmethod
    def get_posembclass(pos_enc_layer_type,selfattention_layer_type):
        #position embedding layer
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        return pos_enc_class

    @staticmethod
    def getattentionMHA(
        selfattention_layer_type,
        pos_enc_layer_type,
        attention_heads,
        output_size,
        attention_dropout_rate):

        if selfattention_layer_type == "selfattn":
            attn_layer = MultiHeadedAttention
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            attn_layer = RelPositionMultiHeadedAttention
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)
        attn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        return attn_layer, attn_layer_args

    @staticmethod
    def get_positionwise_layer(
        attention_dim=256,
        linear_units=2048,
        dropout_rate=0.1,
        activation=None,
    ):
        """Define positionwise layer."""
        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (
            attention_dim, 
            linear_units, 
            dropout_rate,
            activation,
        )
        return positionwise_layer, positionwise_layer_args

    def output_size(self) -> int:
        return self._output_size
     
    def forward(
        self,
        feats: torch.Tensor,
        feats_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate forward propagation.

        Args:
            xs_pad (torch.Tensor): Input tensor (#batch, L, input_size).
            ilens (torch.Tensor): Input length (#batch).
            prev_states (torch.Tensor): Not to be used now.

        Returns:
            torch.Tensor: Output tensor (#batch, L, output_size).
            torch.Tensor: Output length (#batch).
            torch.Tensor: Not to be used now.

        """
        #modal alignment
        # print(feats_lengths)  # tensor([26, 16], device='cuda:0', dtype=torch.int32)
        # print(video_lengths)  # tensor([7, 5], device='cuda:0', dtype=torch.int32)
        # import pdb; pdb.set_trace()
        if self.global_cmvn is not None:
            feats = self.global_cmvn(feats)
        # print(feats.shape)  # torch.Size([2, 26, 80])
        # print(video.shape)  # torch.Size([2, 7, 512])
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        video_masks = masks.clone()

        feats, pos_emb_a, masks = self.embed_a(feats, masks) # 返回的位置编码([1,T,D])
        # print(feats.shape)   # torch.Size([2, 5, 256])  Conv2d下采样到1/4长度
        video, pos_emb_v, video_masks = self.embed_v(video, video_masks)
        # print(video.shape)   # torch.Size([2, 7, 256])
        T_length = min(feats.shape[1], video.shape[1]) # T_length: 5
        feats = feats[:, :T_length, :]  # print(feats.shape) torch.Size([2, 5, 256])
        video = video[:, :T_length, :]  # print(video.shape) torch.Size([2, 7, 256])->torch.Size([2, 5, 256])
        pos_emb_a = pos_emb_a[:, :T_length, :] # print(pos_emb_a.shape) torch.Size([1, 5, 256])
        pos_emb_v = pos_emb_v[:, :T_length, :] # print(pos_emb_v.shape) torch.Size([1, 7, 256])->torch.Size([1, 5, 256])
        # print(masks.shape) # torch.Size([2, 1, 5])
        # print(video_masks.shape) # torch.Size([2, 1, 26])
        video_masks = masks.clone()
        # print(masks.shape) # torch.Size([2, 1, 5])
        # print(video_masks.shape) # torch.Size([2, 1, 5])
        feats = (feats, pos_emb_a)
        
        #vencoder + fusion_encoder
        video_memories = []
        for i in range(self.v_num_blocks):
            # wenet.transformer.encoder_layer的ConformerEncoderLayer:x和pos_emb分别输入
            # 这里调用了self-attention, encoder_layer.py line234
            video, video_masks, _, _ = self.v_encoderlayers[i](video,video_masks,pos_emb_v) 
            # wenet.transformer.encoder_layer_misp2022_avsr的CrossAttentionFusionEncoderLayer中x和video可以为tuple
            # 这里调用了self-attention, encoder_layer_misp2022_avsr.py line140 
            feats, masks, _, _ = self.cross_fusion_encoderlayers[i](feats,masks,(video, pos_emb_v),video_masks)  
            # if isinstance(feats,tuple): #if absemb
            #     video_memories.append(video[0])  
            # else:
            #     video_memories.append(video)
            video_memories.append(video)

        #concat layer
        video_memories = self.vmemory_fusion(torch.cat(video_memories,axis=-1))
        video_memories = self.vmemory_fusion_norm(video_memories)

        #fusion_encoder
        for i in range(self.v_num_blocks,self.crossfusion_num_blocks):
            feats, masks,_,_ = self.cross_fusion_encoderlayers[i](feats,masks,video_memories,video_masks)

        if isinstance(feats, tuple):
            feats = feats[0]
        if self.normalize_before:
            feats = self.after_norm(feats)

        # olens = masks.squeeze(1).sum(1)
        # return feats, olens
        T_min = min(feats.shape[1], masks.shape[2])
        feats = feats[:, :T_min, :]  # torch.Size([2, 21, 256]) 
        masks = masks[:, :, :T_min]  # torch.Size([2, 1, 22])->torch.Size([2, 1, 21])
        return feats, masks