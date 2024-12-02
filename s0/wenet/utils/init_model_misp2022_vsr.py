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
from wenet.transformer.avsr_model_misp2022_vsr import AVSRModel_MISP2022_VSR
from wenet.transformer.ctc_misp2022 import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder

from espnet2.asr.decoder.transformer_decoder import ConvTransformerDecoder
from espnet2.asr.frontend.video_frontend import VideoFrontend

# add
from wenet.branchformer.encoder import BranchformerEncoder
from wenet.e_branchformer.encoder import EBranchformerEncoder 


def init_model_misp2022_vsr(configs):
    """
        MISP2022_VSR[Wenet]: conv3d+resnet18 conformer[Wenet]
    """
    vocab_size = configs['output_dim']

    videofront_type = configs.get('videofront', '')
    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')
    ctc_weight = configs['model_conf']['ctc_weight']

    # 1.video frontend 
    if videofront_type == 'conv3d+resnet18':
        video_frontend = VideoFrontend(**configs['videofront_conf']) #equal to  hidden layer dim
        input_dim = video_frontend.output_size()
    else:
        video_frontend = None
        input_dim = configs['input_dim']

    # 2. Encoder
    if encoder_type == 'conformer':
        encoder = ConformerEncoder(input_dim,
                                   **configs['encoder_conf'])
    elif encoder_type == 'branchformer':
        encoder = BranchformerEncoder(input_dim,
                                      **configs['encoder_conf'])
    elif encoder_type == 'e_branchformer':
        encoder = EBranchformerEncoder(input_dim,
                                       **configs['encoder_conf'])
    elif encoder_type == 'squeezeformer':
        encoder = SqueezeformerEncoder(input_dim,
                                       **configs['encoder_conf'])
    elif encoder_type == 'efficientConformer':
        encoder = EfficientConformerEncoder(input_dim,
                                            **configs['encoder_conf'],
                                            **configs['encoder_conf']['efficient_conf']
                                            if 'efficient_conf' in
                                               configs['encoder_conf'] else {})
    else:
        encoder = TransformerEncoder(input_dim,
                                     **configs['encoder_conf'])

    if decoder_type == 'transformer':
        decoder = TransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    elif decoder_type == 'contransformer':
        # ADD: zhewang18
        decoder = ConvTransformerDecoder(vocab_size, encoder.output_size(),
                                     **configs['decoder_conf'])
    else:
        assert 0.0 < configs['model_conf']['reverse_weight'] < 1.0
        assert configs['decoder_conf']['r_num_blocks'] > 0
        decoder = BiTransformerDecoder(vocab_size, encoder.output_size(),
                                       **configs['decoder_conf'])
    if ctc_weight == 1.0:
        decoder = None

    ctc = CTC(vocab_size, encoder.output_size())

    # Init joint CTC/Attention model
    model = AVSRModel_MISP2022_VSR(vocab_size=vocab_size,
                                   video_frontend=video_frontend,
                                   encoder=encoder,
                                   decoder=decoder,
                                   ctc=ctc,
                                   **configs['model_conf'])
    # for p, pt in model.state_dict().items():
    #     if p.startswith('video_frontend'):
    #         print(p, pt.shape)            
    return model
