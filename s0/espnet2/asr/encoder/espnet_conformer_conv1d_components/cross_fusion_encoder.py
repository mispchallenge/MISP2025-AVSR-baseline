# # """Encoder definition."""
# from .encoder_layer import EncoderLayer,CrossAttentionFusionEncoderLayer
# import logging
# import torch
# from .convolution import ConvolutionModule
# from .encoder_layer import EncoderLayer
# from .getactivate import get_activation
# from .attention import (
#     MultiHeadedAttention,  # noqa: H301
#     RelPositionMultiHeadedAttention,  # noqa: H301
#       # noqa: H301
# )
# from .nets_utils import make_pad_mask
# from .embedding import (
#     PositionalEncoding,  # noqa: H301
#     ScaledPositionalEncoding,  # noqa: H301
#     RelPositionalEncoding,  # noqa: H301
#     LegacyRelPositionalEncoding,  # noqa: H301
# )
# from .layer_norm import LayerNorm
# from .multi_layer_conv import Conv1dLinear
# from .multi_layer_conv import MultiLayeredConv1d
# from .positionwise_feed_forward import (
#     PositionwiseFeedForward,  # noqa: H301
# )
# from .repeat import repeat

# """Encoder definition."""
from .encoder_layer import EncoderLayer,CrossAttentionFusionEncoderLayer
import logging
import torch
from .convolution import ConvolutionModule
from .encoder_layer import EncoderLayer
from .getactivate import get_activation
from .attention import (
    MultiHeadedAttention,  # noqa: H301
    RelPositionMultiHeadedAttention,  # noqa: H301
      # noqa: H301
)
from .nets_utils import make_pad_mask
from .embedding import (
    PositionalEncoding,  # noqa: H301
    ScaledPositionalEncoding,  # noqa: H301
    RelPositionalEncoding,  # noqa: H301
    LegacyRelPositionalEncoding,  # noqa: H301
)
from .layer_norm import LayerNorm
from .multi_layer_conv import Conv1dLinear
from .multi_layer_conv import MultiLayeredConv1d
from .positionwise_feed_forward import (
    PositionwiseFeedForward,  # noqa: H301
)
from .repeat import repeat

class NewDimConvert(torch.nn.Module): #(B,T,D)->(B,T,D)
    def __init__(
        self,
        in_channels:int,
        out_channels:int,
        dropout_rate: float = 0.1,
    ):  
        super().__init__()
        self.convert = torch.nn.Sequential(
            torch.nn.Linear(in_channels,out_channels),
            torch.nn.LayerNorm(out_channels),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout_rate),
            )

    def forward(self,tensor):
        return self.convert(tensor)


def get_posembclass(pos_enc_layer_type,selfattention_layer_type):
        #position embedding layer
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "scaled_abs_pos":
            pos_enc_class = ScaledPositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            assert selfattention_layer_type == "rel_selfattn"
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "legacy_rel_pos":
            assert selfattention_layer_type == "legacy_rel_selfattn"
            pos_enc_class = LegacyRelPositionalEncoding
            logging.warning(
                "Using legacy_rel_pos and it will be deprecated in the future."
            )
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)
        return pos_enc_class

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

"""
AVCrossAttentionEncoder is used for feat vggpreencode 25ps +video 25 ps ,different with other encoder it use audio feat as K, and speech_feat as K,Q, and it will be fine-tune based on a 12-layer asr comferformer
"""
class AVCrossAttentionEncoder(torch.nn.Module): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        v_num_blocks: int = 3,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        crossfusion_num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer="conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        src_first: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        srcattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
    ):  

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
        pos_enc_class = get_posembclass(pos_enc_layer_type,selfattention_layer_type)

        self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )

        
        vencoder_selfattn_layer,vencoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_selfattn_layer,encoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_srcattn_layer,encoder_srcattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(srcattention_layer_type,
                                                                                pos_enc_layer_type,
                                                                                attention_heads,
                                                                                output_size,
                                                                                zero_triu,
                                                                                attention_dropout_rate)

        self.normalize_before = normalize_before
        if positionwise_layer_type == "linear":
            positionwise_layer = PositionwiseFeedForward
            positionwise_layer_args = (
                output_size,
                linear_units,
                dropout_rate,
                activation,
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

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)
        
        #video branch
        if v_num_blocks !=0:
            self.v_encoder = repeat(
                v_num_blocks,
                lambda lnum: EncoderLayer(
                    output_size,
                    vencoder_selfattn_layer(*vencoder_selfattn_layer_args),
                    positionwise_layer(*positionwise_layer_args),
                    positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                    convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                    dropout_rate,
                    normalize_before,
                    concat_after,
                ),
            )
        else: 
            self.v_encoder = None
            self.video_norm = LayerNorm(output_size)

        #audio branch
        self.cross_fusion_encoder = torch.nn.ModuleList()
        for i in range(crossfusion_num_blocks):
            self.cross_fusion_encoder.append(
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
                    )
            )
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size
    
    @staticmethod
    def getattentionMHA(
        selfattention_layer_type,
        pos_enc_layer_type,
        attention_heads,
        output_size,
        zero_triu,
        attention_dropout_rate):

        if selfattention_layer_type == "selfattn":
            attn_layer = MultiHeadedAttention
            attn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
            )
        elif selfattention_layer_type == "rel_selfattn":
            assert pos_enc_layer_type == "rel_pos"
            attn_layer = RelPositionMultiHeadedAttention
            attn_layer_args = (
                attention_heads,
                output_size,
                attention_dropout_rate,
                zero_triu,
            )
        else:
            raise ValueError("unknown encoder_attn_layer: " + selfattention_layer_type)
        return attn_layer, attn_layer_args
     
    def forward(self,feats,feats_lengths,video,video_lengths
        ):
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
        feats_lengths = feats_lengths.min(video_lengths)
        feats = feats[:,:max(feats_lengths)] #B,T,512
        video = video[:,:max(feats_lengths)] #B,T,512
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        video_masks = masks.clone()

     
        att_augmasks = [None]*len(self.cross_fusion_encoder)

        #posemb
        feats = self.embed(feats) #add posemb
        if not isinstance(feats,tuple): #if absemb
            video = self.embed(video)
        else:
            video = (video,feats[1])
        
        #vencoder
        if self.v_encoder:
            video,video_masks = self.v_encoder(video,video_masks)
        else:
            if isinstance(feats,tuple): #if relemb
                video = video[0]
            video = self.video_norm(video)
    
        #crossfusionencoder
        for cross_fusion_encoder,att_augmask in zip(self.cross_fusion_encoder,att_augmasks):
            feats,masks,video,video_masks = cross_fusion_encoder(feats,masks,video,video_masks,att_augmask)

        if isinstance(feats, tuple):
            feats = feats[0]
        if self.normalize_before:
            feats = self.after_norm(feats)

        olens = masks.squeeze(1).sum(1)
        return feats, olens, None

"""
NewAVCrossAttentionEncoder is used for feat vggpreencode 25ps +video 25 ps ,different with other encoder it use audio feat as K, and speech_feat as K,Q, and it will be fine-tune based on a 12-layer asr comferformer
"""
class NewAVCrossAttentionEncoder(torch.nn.Module): # [b,T,512]->[b,T,256]
    def __init__(
        self,
        v_num_blocks: int = 3,
        output_size: int = 256,
        attention_heads: int = 4,
        linear_units: int = 2048,
        crossfusion_num_blocks: int = 12,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        input_layer="conv2d",
        normalize_before: bool = True,
        concat_after: bool = False,
        src_first: bool = False,
        positionwise_layer_type: str = "linear",
        positionwise_conv_kernel_size: int = 3,
        macaron_style: bool = False,
        rel_pos_type: str = "legacy",
        pos_enc_layer_type: str = "rel_pos",
        selfattention_layer_type: str = "rel_selfattn",
        srcattention_layer_type: str = "rel_selfattn",
        activation_type: str = "swish",
        use_cnn_module: bool = True,
        zero_triu: bool = False,
        cnn_module_kernel: int = 31,
    ):  
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
        pos_enc_class = get_posembclass(pos_enc_layer_type,selfattention_layer_type)

        self.embed = torch.nn.Sequential(
                pos_enc_class(output_size, positional_dropout_rate)
            )

        
        vencoder_selfattn_layer,vencoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_selfattn_layer,encoder_selfattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(selfattention_layer_type,
                                                                                    pos_enc_layer_type,
                                                                                    attention_heads,
                                                                                    output_size,
                                                                                    zero_triu,
                                                                                    attention_dropout_rate)
        encoder_srcattn_layer,encoder_srcattn_layer_args = AVCrossAttentionEncoder.getattentionMHA(srcattention_layer_type,
                                                                                pos_enc_layer_type,
                                                                                attention_heads,
                                                                                output_size,
                                                                                zero_triu,
                                                                                attention_dropout_rate)

        self.normalize_before = normalize_before
        positionwise_layer,positionwise_layer_args = get_positionwise_layer(positionwise_layer_type,
                                                                                                        output_size,
                                                                                                        linear_units,
                                                                                                        dropout_rate,
                                                                                                        positionwise_conv_kernel_size)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation)
        
        self.v_encoderlayers = torch.nn.ModuleList()
        for _ in range(v_num_blocks):
            self.v_encoderlayers.append(
                EncoderLayer(
                        output_size,
                        vencoder_selfattn_layer(*vencoder_selfattn_layer_args),
                        positionwise_layer(*positionwise_layer_args),
                        positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                        convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                        dropout_rate,
                        normalize_before,
                        concat_after,
                    ),
                )    

        self.vmemory_fusion = NewDimConvert(in_channels=output_size*v_num_blocks,out_channels=output_size)  
        self.vmemory_fusion_norm = LayerNorm(output_size)    

        self.cross_fusion_encoderlayers = torch.nn.ModuleList()
        for _ in range(crossfusion_num_blocks):
            self.cross_fusion_encoderlayers.append(
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
                src_first
            ),)   
    
        if self.normalize_before:
            self.after_norm = LayerNorm(output_size)

    def output_size(self) -> int:
        return self._output_size


     
    def forward(self,feats,feats_lengths,video,video_lengths,
        ):
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
        feats_lengths = feats_lengths.min(video_lengths)
        feats = feats[:,:max(feats_lengths)] #B,T,512
        video = video[:,:max(feats_lengths)] #B,T,512
        masks = (~make_pad_mask(feats_lengths)[:, None, :]).to(feats.device) 
        video_masks = masks.clone()

        #posemb
        feats = self.embed(feats) #add posemb
        if not isinstance(feats,tuple): #if absemb
            video = self.embed(video)
        else:
            video = (video,feats[1])
        
        #vencoder+fusion_encoder
        video_memories = []
        for i in range(self.v_num_blocks):
            video,video_masks = self.v_encoderlayers[i](video,video_masks)
            feats,masks,_,_ = self.cross_fusion_encoderlayers[i](feats,masks,video,video_masks)
            if isinstance(feats,tuple): #if absemb
                video_memories.append(video[0])  
            else:
                video_memories.append(video)

        #concat layer
        video_memories = self.vmemory_fusion(torch.cat(video_memories,axis=-1))
        video_memories = self.vmemory_fusion_norm(video_memories)

        #fusion_encoder
        for i in range(self.v_num_blocks,self.crossfusion_num_blocks):
            feats,masks,_,_ = self.cross_fusion_encoderlayers[i](feats,masks,video_memories,video_masks)

        if isinstance(feats, tuple):
            feats = feats[0]
        if self.normalize_before:
            feats = self.after_norm(feats)

        olens = masks.squeeze(1).sum(1)
        return feats, olens, None


if __name__ == "__main__":
    yaml_path = "/yrfs2/cv1/hangchen2/espnet/mispi/avsr/conf/avsrfinetune/avsr_crosscom_finetune.yaml"
    import yaml 
    with open(yaml_path) as f :
        cfg = yaml.safe_load(f)
    encoder_conf = cfg["encoder_conf"]
    encoder = NewAVCrossAttentionEncoder(**encoder_conf)
    print(encoder)
    feats_lengths = torch.randint(128,129,(32,))                                  
    feats = torch.rand(32,128,512)
    video_lengths = torch.randint(128,129,(32,))     
    video = torch.rand(32,128,512)
    output,olen,_ = encoder(feats,feats_lengths,video,video_lengths)
    print(output.shape,olen)

