# Copyright (c) 2023 USTC (Zhe Wang)
# 此代码为复现ICME论文asr模型
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
from wenet.transformer.asr_model_misp2022_asr import ASRModel_MISP2022_ASR
from wenet.transformer.cmvn import GlobalCMVN
from wenet.transformer.ctc_misp2022 import CTC
from wenet.transformer.decoder import BiTransformerDecoder, TransformerDecoder
from wenet.transformer.encoder import ConformerEncoder, TransformerEncoder
from wenet.squeezeformer.encoder import SqueezeformerEncoder
from wenet.efficient_conformer.encoder import EfficientConformerEncoder
from wenet.utils.cmvn import load_cmvn

from espnet2.asr.decoder.transformer_decoder import ConvTransformerDecoder
from espnet2.asr.preencoder.wav import VGGfeatPreEncoder


def init_model_misp2022_asr(configs):
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

    input_dim = configs['input_dim']
    vocab_size = configs['output_dim']

    preencoder_type = configs.get('preencoder','')
    encoder_type = configs.get('encoder', 'conformer')
    decoder_type = configs.get('decoder', 'bitransformer')

    if preencoder_type == 'vggfeat':
        preencoder = VGGfeatPreEncoder(**configs['preencoder_conf'])  #[B,T]->[B,T,D] 25fps
    else:
        preencoder = None

    if encoder_type == 'conformer':
        # encoder = ConformerEncoder(input_dim,
        #                            global_cmvn=global_cmvn,
        #                            **configs['encoder_conf'])
        encoder = ConformerEncoder(input_dim,
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
        # model = ASRModel_MISP2022(vocab_size=vocab_size,
        #                           encoder=encoder,
        #                           decoder=decoder,
        #                           ctc=ctc,
        #                           **configs['model_conf'])
        model = ASRModel_MISP2022_ASR(vocab_size=vocab_size,
                                      global_cmvn=global_cmvn,
                                      preencoder=preencoder,
                                      encoder=encoder,
                                      decoder=decoder,
                                      ctc=ctc,
                                      **configs['model_conf'])
    return model
