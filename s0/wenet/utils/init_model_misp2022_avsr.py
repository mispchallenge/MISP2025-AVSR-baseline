# Copyright (c) 2023 USTC (Zhe Wang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from wenet.transducer.joint import TransducerJoint
from wenet.transducer.predictor import (ConvPredictor, EmbeddingPredictor,
                                        RNNPredictor)
from wenet.transducer.transducer import Transducer
from wenet.transformer.avsr_model_misp2022_avsr import AVSRModel_MISP2022_AVSR
from espnet2.asr.frontend.video_frontend import VideoFrontend
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc_misp2022 import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder_misp2022_avsr import NewAVCrossAttentionEncoder

from wenet.utils.cmvn import load_cmvn

# add
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.e_branchformer.encoder import EBranchformerEncoder  


def init_model_misp2022_avsr(configs):
    """
        MISP2022_ASR[Wenet]: conformer[Wenet]+contransformer
    """
    # MISP2022_ASR[Espnet]: 提特征->SpecAug->CMVN->Preencoder->Encoder
    # Wenet中: ConformerEncoder有CMVN这个选项。
    # Aishell中ConformerEncoder传入global_cmvn参数,不含Preencoder模块。
    # MISP2022_ASR[Wenet]中ConformerEncoder[Wenet]不传入global_cmvn参数,包含Preencoder模块。
    if configs['cmvn_file'] is not None:
        mean, istd = load_cmvn(configs['cmvn_file'], configs['is_json_cmvn'])
        global_cmvn = GlobalCMVN(
            torch.from_numpy(mean).float(),
            torch.from_numpy(istd).float())
    else:
        global_cmvn = None

    input_dim_a = configs['input_dim_a']
    vocab_size = configs['output_dim']

    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')
    videofront_type = configs.get('videofront', '')

    # 1.video frontend 
    if videofront_type == 'conv3d+resnet18':
        video_frontend = VideoFrontend(**configs['videofront_conf']) #equal to  hidden layer dim
        input_dim_v = video_frontend.output_size()
    else:
        video_frontend = None
        input_dim_v = configs['input_dim']

    assert encoder_type == 'AVconformer'
    if encoder_type == 'AVconformer':
        encoder = NewAVCrossAttentionEncoder(input_size_a=input_dim_a, input_size_v=input_dim_v, global_cmvn=global_cmvn, **configs['encoder_conf'])
    elif encoder_type == 'AVbranchformer':
        encoder = None
    elif encoder_type == 'AVe_branchformer':
        encoder = None

    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    ctc = CTC(vocab_size, encoder.output_size())

    # Init joint CTC/Attention or Transducer model
    if 'predictor' in configs:
        predictor_type = configs.get('predictor', 'rnn')
        if predictor_type == 'rnn':
            predictor = RNNPredictor(vocab_size, **configs['predictor_conf'])
        elif predictor_type == 'embedding':
            predictor = EmbeddingPredictor(vocab_size,
                                           **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        elif predictor_type == 'conv':
            predictor = ConvPredictor(vocab_size, **configs['predictor_conf'])
            configs['predictor_conf']['output_size'] = configs[
                'predictor_conf']['embed_size']
        else:
            raise NotImplementedError(
                "only rnn, embedding and conv type support now")
        configs['joint_conf']['enc_output_size'] = configs['encoder_conf'][
            'output_size']
        configs['joint_conf']['pred_output_size'] = configs['predictor_conf'][
            'output_size']
        joint = TransducerJoint(vocab_size, **configs['joint_conf'])
        model = Transducer(vocab_size=vocab_size,
                           blank=0,
                           predictor=predictor,
                           encoder=encoder,
                           attention_decoder=decoder,
                           joint=joint,
                           ctc=ctc,
                           **configs['model_conf'])
    else:
        # Init joint CTC/Attention model
        model = AVSRModel_MISP2022_AVSR(vocab_size=vocab_size,
                                        video_frontend=video_frontend,
                                        encoder=encoder,
                                        decoder=decoder,
                                        ctc=ctc,
                                        **configs['model_conf'])
    for p, pt in model.state_dict().items():
        if p.startswith('decoder'):
            print(p, pt.shape)  

    return model
