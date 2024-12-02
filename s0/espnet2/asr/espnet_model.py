from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types
import torch.nn as nn
from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder,AVInDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder,AVOutEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import get_transducer_task_io
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet2.asr.frontend.video_frontend import VideoFrontend
from espnet2.asr.preencoder.wav import WavPreEncoder
from espnet2.asr.preencoder.wav import featPreEncoder
if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
        only_pdfloss: bool = False,
        pdfloss_skipencoder: bool = False,
        pdfloss_weight: float = 0.0,
        pdf_lsm_weigth: float = 0.0,
        pdf_cnum: int = 9024,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight+pdfloss_weight <= 1.0, f"ctc:{ctc_weight},pdf:{pdfloss_weight}"

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()
        self.only_pdfloss = only_pdfloss
        self.pdfloss_skipencoder = pdfloss_skipencoder
        self.pdfloss_weight = pdfloss_weight
        self.pdf_cnum = pdf_cnum

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if abs(1.0-self.ctc_weight-self.pdfloss_weight) <= 1e-5 or only_pdfloss == True:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0 or only_pdfloss == True:
            self.ctc = None
        else:
            self.ctc = ctc

        if pdfloss_weight != 0.0 or only_pdfloss==True :
            if not pdfloss_skipencoder:
                self.pdfclass_linear  = torch.nn.Linear(encoder.output_size(), pdf_cnum)
            else: 
                self.pdfclass_linear  = torch.nn.Linear(frontend.output_size(), pdf_cnum)
            self.criterion_pdf = LabelSmoothingLoss(
                size=pdf_cnum,
                padding_idx=ignore_id,
                smoothing=pdf_lsm_weigth,
            )

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        pdf: torch.Tensor=None,
        pdf_lengths: torch.Tensor=None,
        **kwargs,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        if pdf != None :
            assert pdf.shape[0] == pdf_lengths.shape[0] == text.shape[0]
            pdf = pdf[:, : pdf_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens, frontend_out, frontend_out_lens = self.encode(speech, speech_lengths)     
        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()
        # 2. Loss
        # 2.1. only_pdfloss
        if self.only_pdfloss:
            assert pdf != None, "pdf_weight:{self.pdfloss_weight} or check pdf input"
            loss,acc_pdf = self._calc_pdf_loss(
                frontend_out if self.pdfloss_skipencoder else encoder_out, pdf,
                )
          
            stats["loss_pdf"] = loss.detach() if loss is not None else None
            stats["acc_pdf"] = acc_pdf
        else:  
        # 2.2. CTC loss
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc

            if self.use_transducer_decoder:
                # 2.3. Trasducer loss
                (
                    loss_transducer,
                    cer_transducer,
                    wer_transducer,
                ) = self._calc_transducer_loss(
                    encoder_out,
                    encoder_out_lens,
                    text,
                )

                if loss_ctc is not None:
                    loss = loss_transducer + (self.ctc_weight * loss_ctc)
                else:
                    loss = loss_transducer

                # Collect Transducer branch stats
                stats["loss_transducer"] = (
                    loss_transducer.detach() if loss_transducer is not None else None
                )
                stats["cer_transducer"] = cer_transducer
                stats["wer_transducer"] = wer_transducer

            else:
                # 2.4. pdf loss 
                if self.pdfloss_weight != 0.0:
                    assert pdf != None, "check pdf input"
                    loss_pdf,acc_pdf = self._calc_pdf_loss(
                        frontend_out if self.pdfloss_skipencoder else encoder_out, pdf,
                    )
                    stats["loss_pdf"] = loss_pdf.detach() if loss_pdf is not None else None
                    stats["acc_pdf"] = acc_pdf

                # 2.5. Attention loss
                if abs(1.0-self.ctc_weight-self.pdfloss_weight) >= 1e-5:
                    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )
                    # Collect Attn branch stats
                    stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                    stats["acc"] = acc_att
                    stats["cer"] = cer_att
                    stats["wer"] = wer_att

                #  2.6. weighted sum loss
                if self.pdfloss_weight == 1.0:
                    loss = loss_pdf
                elif self.ctc_weight == 1.0:
                    loss = loss_ctc
                elif self.pdfloss_weight == 0. and self.ctc_weight == 0.:
                    loss = loss_att
                elif self.pdfloss_weight == 0.0:
                    loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
                elif self.ctc_weight == 0.0:
                    loss = self.pdfloss_weight * loss_pdf + (1 - self.pdfloss_weight) * loss_att
                elif abs(self.pdfloss_weight + self.ctc_weight-1.0) <= 1e-5 :
                    loss = self.ctc_weight * loss_ctc + self.pdfloss_weight * loss_pdf
                else:
                    loss = self.ctc_weight * loss_ctc + self.pdfloss_weight * loss_pdf + (1.0-self.pdfloss_weight-self.ctc_weight) * loss_att

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        if not isinstance (self.preencoder,WavPreEncoder):
            with autocast(False):
                # 1. Extract feats
                feats, feats_lengths = self._extract_feats(speech, speech_lengths)

                # 2. Data augmentation
                if self.specaug is not None and self.training:
                    feats, feats_lengths = self.specaug(feats, feats_lengths)

                # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)

        # 4. Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            if isinstance (self.preencoder,WavPreEncoder):
                feats, feats_lengths = self.preencoder(speech, speech_lengths)  #[B,T]->[B,T,D] 25fps
            else:
                feats, feats_lengths = self.preencoder(feats, feats_lengths) #[B,T,D]->[B,T,D] 100fps

        # 5. Forward encoder (Batch, Length, Dim) -> (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        # 6. Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens, feats, feats_lengths

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for multiple channels #B,T,channel_num-> channel_num,B,T,->channel_num*B,T
        if speech.dim() == 3:
            bsize,tlen,channel_num = speech.shape
            speech = speech.permute((2, 0, 1)).reshape(channel_num*bsize,tlen) 
            speech_lengths = speech_lengths.repeat(channel_num)

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_pdf_loss(
        self,
        encoder_out: torch.Tensor,
        ys_pad: torch.Tensor,
    ):  
        ys_head_pd = self.pdfclass_linear(encoder_out) #B,T,encodeoutdim -> B,T,class num
        
        #soft align for some pdf and label
        tag_len = ys_pad.shape[1]
        hyp_len = ys_head_pd.shape[1]
        if  tag_len != hyp_len:
            if abs(tag_len-hyp_len) / min(tag_len,hyp_len) > 0.1:
                assert f"ys_pad_shape:{ys_pad.shape},ys_head_pad_shape:{ys_head_pd.shape}"
            else:
                cutlen = min(tag_len,hyp_len)
                ys_head_pd = ys_head_pd[:,:cutlen] #[B,T,C]
                ys_pad = ys_pad[:,:cutlen] #[B,T]

        loss_pdf = self.criterion_pdf(ys_head_pd, ys_pad) #ignore_id = -1
        
        acc_pdf = th_accuracy(
                ys_head_pd.reshape(-1, self.pdf_cnum),
                ys_pad,
                ignore_label=self.ignore_id,
            )
        return loss_pdf,acc_pdf

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):  
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        decoder_in, target, t_len, u_len = get_transducer_task_io(
            labels,
            encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        joint_out = self.joint_network(
            encoder_out.unsqueeze(2), decoder_out.unsqueeze(1)
        )

        loss_transducer = self.criterion_transducer(
            joint_out,
            target,
            t_len,
            u_len,
        )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return loss_transducer, cer_transducer, wer_transducer


class ESPnetAVSRModel(ESPnetASRModel):  
    
    def __init__(
                self,
                vocab_size: int,
                token_list: Union[Tuple[str, ...], List[str]],
                frontend: Optional[AbsFrontend],
                video_frontend : VideoFrontend,
                specaug: Optional[AbsSpecAug],
                normalize: Optional[AbsNormalize],
                preencoder: Optional[AbsPreEncoder],
                encoder: AbsEncoder,
                postencoder: Optional[AbsPostEncoder],
                decoder: AbsDecoder,
                ctc: CTC,
                joint_network: Optional[torch.nn.Module],
                ctc_weight: float = 0.5,
                ignore_id: int = -1,
                lsm_weight: float = 0.0,
                length_normalized_loss: bool = False,
                report_cer: bool = True,
                report_wer: bool = True,
                sym_space: str = "<space>",
                sym_blank: str = "<blank>",
                extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super(ESPnetASRModel, self).__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.video_frontend = video_frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def encode(
        self, 
        speech: torch.Tensor, 
        speech_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """

        #1. video pre-encoder  (B, T, 96, 96,3) -> (B,T,D)  25ps 
        if self.video_frontend is not None:
            video,video_lengths = self.video_frontend(video,video_lengths)

        #2. STFT+AG+Norm
        #WavPreEncoder is 1-convd and resnet 1d based N,T->N,T,C
        #WavPreEncoder only onput accept waveform , don't have to do STSF 
        if not isinstance (self.preencoder,WavPreEncoder):
            with autocast(False):
                # a. Extract feats
                feats, feats_lengths = self._extract_feats(speech, speech_lengths)
                # b. Data augmentation
                if self.specaug is not None and self.training:
                        feats, feats_lengths = self.specaug(feats, feats_lengths)
                
                # c. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                if self.normalize is not None:
                    feats, feats_lengths = self.normalize(feats, feats_lengths)
                    
                    
        #3. Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            if isinstance (self.preencoder,WavPreEncoder):
                feats, feats_lengths = self.preencoder(speech, speech_lengths)  #[B,T]->[B,T,D] 25fps
            else:
                feats, feats_lengths = self.preencoder(feats, feats_lengths) #[B,T,D]->[B,T,D] 100fps
              
        # 4. soft alignment for WavPreEncoder 
        # for wav preencoder auido 25ps, video 25ps; for feat preencoder audio 100 ps ,video 25 ps 
        # align av frames if both audio and video nearly close to 25ps or alignment , if 4*video = audio don't alignment
        if isinstance (self.preencoder,WavPreEncoder):
            if not feats_lengths.equal(video_lengths):
                if (feats_lengths-video_lengths).abs().sum() < (feats_lengths-video_lengths*4).abs().sum():
                    feats_lengths = feats_lengths.min(video_lengths)
                    video_lengths = feats_lengths.clone()
                    feats = feats[:,:max(feats_lengths)]
                    video = video[:,:max(feats_lengths)]
        
        # 5. Encoder
        # 5.1. for encoders which only output audio memories
        if not isinstance(self.encoder,AVOutEncoder):
            encoder_out, encoder_out_lens, _ = self.encoder(feats,feats_lengths,video,video_lengths)#[B,T,D]
    
            # Post-encoder, e.g. NLU
            if self.postencoder is not None:
                encoder_out, encoder_out_lens = self.postencoder(
                    encoder_out, encoder_out_lens
                )

            assert encoder_out.size(0) == speech.size(0), (
                encoder_out.size(),
                speech.size(0),
            )
            assert encoder_out.size(1) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

            return encoder_out, encoder_out_lens
        
        # 5.2. for encoders which output audio and video memories
        else:
            feats,feats_lengths,video,video_lengths,_ = self.encoder(feats,feats_lengths,video,video_lengths)#[B,T,D]
            assert feats.size(0)==video.size(0)==speech.size(0), (
                    feats.size(),
                    video.size(),
                    speech.size(0),
                )
            assert feats.size(1) <= feats_lengths.max() and video.size(1) <= video_lengths.max(), (
                feats.size(),
                feats_lengths.max(),
                video.size(),
                video_lengths.max(),
            )

            return feats,feats_lengths,video,video_lengths

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == video.shape[0]
            == video_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape,video.shape,text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        if not isinstance(self.encoder,AVOutEncoder): 
            encoder_out, encoder_out_lens = self.encode(speech, speech_lengths,video,video_lengths)
        else:
            encoder_out,encoder_out_lens,encoder_vout,encoder_vout_lens = self.encode(speech, speech_lengths,video,video_lengths)
        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()

        # 2. loss
        # 2.1. CTC loss
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
                # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        if self.use_transducer_decoder:
            # 2.2. Transducer loss
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2.3. Attention loss
            if self.ctc_weight != 1.0:
                if not isinstance(self.decoder,AVInDecoder):
                    loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                        encoder_out, encoder_out_lens, text, text_lengths
                    )
                else:
                    loss_att, acc_att, cer_att, wer_att = self._calc_avin_att_loss(
                        encoder_out, encoder_out_lens,encoder_vout,encoder_vout_lens,text, text_lengths
                    )

            # 2.4. CTC-Att loss 
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att
            
            

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    # for decoder input audio and video memeories
    def _calc_avin_att_loss(
        self,
        feats: torch.Tensor,
        feats_lens: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):  
    
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            feats, feats_lens, video, video_lengths, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    
class ESPnetVSRModel(ESPnetASRModel):  
    
    def __init__(
                self,
                vocab_size: int,
                token_list: Union[Tuple[str, ...], List[str]],
                video_frontend : VideoFrontend,
                encoder: AbsEncoder,
                decoder: AbsDecoder,
                ctc: CTC,
                joint_network: Optional[torch.nn.Module],
                ctc_weight: float = 0.5,
                ignore_id: int = -1,
                lsm_weight: float = 0.0,
                length_normalized_loss: bool = False,
                report_cer: bool = True,
                report_wer: bool = True,
                sym_space: str = "<space>",
                sym_blank: str = "<blank>",
                extract_feats_in_collect_stats: bool = True,
                only_pdfloss: bool = False,
                pdfloss_skipencoder: bool = False,
                pdfloss_weight: float = 0.0,
                pdf_lsm_weigth: float = 0.0,
                pdf_cnum: int = 9024,

    ): 
        assert check_argument_types()
        assert 0.0 <= ctc_weight+pdfloss_weight <= 1.0, f"ctc:{ctc_weight},pdf:{pdfloss_weight}"
         
        super(ESPnetASRModel, self).__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()
        self.only_pdfloss = only_pdfloss
        self.pdfloss_skipencoder = pdfloss_skipencoder
        self.pdfloss_weight = pdfloss_weight
        self.pdf_cnum = pdf_cnum

        self.video_frontend = video_frontend
        self.encoder = encoder

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        
        # we set self.decoder = None in the CTC mode since
        # self.decoder parameters were never used and PyTorch complained
        # and threw an Exception in the multi-GPU experiment.
        # thanks Jeff Farris for pointing out the issue.
        if abs(1.0-self.ctc_weight-self.pdfloss_weight) <= 1e-5 or only_pdfloss == True:
            self.decoder = None
        else:
            self.decoder = decoder
            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )


        if report_cer or report_wer:
            self.error_calculator = ErrorCalculator(
                token_list, sym_space, sym_blank, report_cer, report_wer
            )

        if ctc_weight == 0.0 or only_pdfloss == True:
            self.ctc = None
        else:
            self.ctc = ctc
        
        if pdfloss_weight != 0.0 or only_pdfloss==True :
            if not pdfloss_skipencoder:
                self.pdfclass_linear  = torch.nn.Linear(encoder.output_size(), pdf_cnum)
            else: 
                self.pdfclass_linear  = torch.nn.Linear(video_frontend.output_size(), pdf_cnum)
            self.criterion_pdf = LabelSmoothingLoss(
                size=pdf_cnum,
                padding_idx=ignore_id,
                smoothing=pdf_lsm_weigth,
            )

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats
    
    def encode(
        self, 
        video: torch.Tensor,
        video_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        #video pre-encoder (B, T, 96, 96,3) -> (B,T,D)  25ps 
        video,video_lengths = self.video_frontend(video,video_lengths)
        encoder_res = self.encoder(video,video_lengths)#[B,T,D]
        encoder_out, encoder_out_lens = encoder_res[0],encoder_res[1]
    
        assert encoder_out.size(0) == video.size(0), (
            encoder_out.size(),
            video.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens,video,video_lengths

    def forward(
        self,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
        pdf: torch.Tensor=None,
        pdf_lengths: torch.Tensor=None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """ 
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            text.shape[0]
            == text_lengths.shape[0]
            == video.shape[0]
            == video_lengths.shape[0]
        ), (text.shape, text_lengths.shape,video.shape,text_lengths.shape)
        batch_size = video.shape[0]
        

        # for data-parallel
        text = text[:, : text_lengths.max()]

        if pdf != None :
            assert pdf.shape[0] == pdf_lengths.shape[0] == text.shape[0]
            pdf = pdf[:, : pdf_lengths.max()]

   
        # 1. Encoder
        encoder_out, encoder_out_lens, frontend_out, frontend_out_lens  = self.encode(video,video_lengths)
        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_pdf, acc_pdf = None, None
        stats = dict()
        # 2. loss
        # 2.1. only_pdfloss
        if self.only_pdfloss:
            assert pdf != None, "pdf_weight:{self.pdfloss_weight} or check pdf input"
            loss,acc_pdf = self._calc_pdf_loss(
                frontend_out if self.pdfloss_skipencoder else encoder_out, pdf,
                )
            stats["loss_pdf"] = loss.detach() if loss is not None else None
            stats["acc_pdf"] = acc_pdf
         
        else:
            # 2.2. ctc loss
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
                # Collect CTC branch stats
                stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
                stats["cer_ctc"] = cer_ctc

            # 2.2. pdf loss
            if self.pdfloss_weight != 0.0:
                assert pdf != None, "check pdf input"
                loss_pdf,acc_pdf = self._calc_pdf_loss(
                    frontend_out if self.pdfloss_skipencoder else encoder_out, pdf,
                )
                stats["loss_pdf"] = loss_pdf.detach() if loss_pdf is not None else None
                stats["acc_pdf"] = acc_pdf

            # 2.3. attention loss
            if abs(1.0-self.ctc_weight-self.pdfloss_weight) >= 1e-5:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
                # Collect Attention branch stats
                stats["loss_att"] = loss_att.detach() if loss_att is not None else None
                stats["acc"] = acc_att
                stats["cer"] = cer_att
                stats["wer"] = wer_att

            # 2.4. weight sum loss
            if self.pdfloss_weight == 1.0:
                loss = loss_pdf
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            elif self.pdfloss_weight == 0. and self.ctc_weight == 0.:
                loss = loss_att
            elif self.pdfloss_weight == 0.0:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att
            elif self.ctc_weight == 0.0:
                loss = self.pdfloss_weight * loss_pdf + (1 - self.pdfloss_weight) * loss_att
            elif abs(self.pdfloss_weight + self.ctc_weight-1.0) <= 1e-5 :
                loss = self.ctc_weight * loss_ctc + self.pdfloss_weight * loss_pdf
            else:
                loss = self.ctc_weight * loss_ctc + self.pdfloss_weight * loss_pdf + (1.0-self.pdfloss_weight-self.ctc_weight) * loss_att
                
            

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight


class ESPnetChannelsAVSRModel(ESPnetASRModel):  
    
    def __init__(
                self,
                vocab_size: int,
                token_list: Union[Tuple[str, ...], List[str]],
                frontend: Optional[AbsFrontend],
                video_frontend : VideoFrontend,
                specaug: Optional[AbsSpecAug],
                gss_normalize: Optional[AbsNormalize],
                channels_normalize: Optional[AbsNormalize],
                preencoder: Optional[AbsPreEncoder],
                encoder: AbsEncoder,
                postencoder: Optional[AbsPostEncoder],
                decoder: AbsDecoder,
                ctc: CTC,
                joint_network: Optional[torch.nn.Module],
                ctc_weight: float = 0.5,
                ignore_id: int = -1,
                lsm_weight: float = 0.0,
                length_normalized_loss: bool = False,
                report_cer: bool = True,
                report_wer: bool = True,
                sym_space: str = "<space>",
                sym_blank: str = "<blank>",
                extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super(ESPnetASRModel, self).__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.video_frontend = video_frontend
        self.specaug = specaug
        self.gss_normalize = gss_normalize
        self.channels_normalize = channels_normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.use_transducer_decoder = joint_network is not None

        self.error_calculator = None

        if self.use_transducer_decoder:
            from warprnnt_pytorch import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
                fastemit_lambda=0.0,
            )

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        if ctc_weight == 0.0:
            self.ctc = None
        else:
            self.ctc = ctc

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    
    def encode(
        self, 
        speech: torch.Tensor,  #(B, T)
        speech_lengths: torch.Tensor,
        video: torch.Tensor, #(B, T, 96, 96,3) 
        video_lengths: torch.Tensor,
        channels: torch.Tensor, #B,T,channel_nums
        channels_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py
        
        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        # (B, T, 96, 96,3) -> (B,T,D)  25ps 
        video,video_lengths = self.video_frontend(video,video_lengths)
        batch_size,video_T,video_D = video.size()
        #params in WavPreEncoder must be trained, data fromat is wav not spec
        #WavPreEncoder is 1-convd and resnet 1d based N,T->N,T,C
        if not isinstance (self.preencoder,WavPreEncoder):
            # print("-------------- before to frequent domain---------------",file=log_file)
            # print(f"speech:{speech.shape} speech_lengths:{speech_lengths},channels:{channels.shape},channels_lengths:{channels_lengths}",file=log_file)
            with autocast(False):
                # 1. Extract feats
                feats, feats_lengths = self._extract_feats(speech, speech_lengths) #B,channel_num,T,80
                batch_size,_,channel_num = channels.shape
                channels_feats, channels_feats_lengths = self._extract_feats(channels, channels_lengths) # B,T,channel_num -> channel_num*B,T,80  (length=C*T)
               
                # 2. Data augmentation
                if self.specaug is not None and self.training:
                    feats, feats_lengths = self.specaug(feats, feats_lengths)

                # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
                 
                if self.gss_normalize is not None:
                    feats, feats_lengths = self.gss_normalize(feats, feats_lengths)
                if self.channels_normalize is not None:
                    channels_feats, channels_feats_lengths = self.channels_normalize(channels_feats, channels_feats_lengths) # channel_num*B,T,80
                    feats_lengths = feats_lengths.min(channels_feats_lengths[:batch_size])
                    # print("------check lengths------")
                    # print(f"channels_feats_lengths: {channels_feats_lengths[:batch_size]},feats_lengths:{feats_lengths} ",file=log_file)
                    # print("use spectru augmentation and global normalize ",file = log_file)
                    
        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            if isinstance (self.preencoder,WavPreEncoder):
                # logging.info(speech.shape, speech_lengths)
                feats, feats_lengths = self.preencoder(speech, speech_lengths)  #[B,T]->[B,T,D] 25fps
            else:
                feats, feats_lengths = self.preencoder(feats, feats_lengths) #[B,T,D]->[B,T,D] 100fps
                #warnin I decide to subsampling audio 100 ps ->25 ps,because the ctc sequence is too long  
                # upsampling video 25fps->100fps 
                # video = torch.stack([video for _ in range(4)], dim=-1).reshape(B, video_T*4, video_D)
                # video_lengths = video_lengths*4
        # print("--------------after audio and video frontend--------------",file=log_file)
        # print(f"video:{video.shape} length:{video_lengths},feat:{feats.shape},length{feats_lengths}",file=log_file)

         #aligment if both audio and video nearly close to 25ps or alignment , if 4*video = audio don't alignment
         # for wav preencoder 25ps, video 25ps , for feat preencoder audio 100 ps ,video 25 ps 
        if isinstance (self.preencoder,WavPreEncoder):
            if not feats_lengths.equal(video_lengths):
                if (feats_lengths-video_lengths).abs().sum() < (feats_lengths-video_lengths*4).abs().sum():
                    feats_lengths = feats_lengths.min(video_lengths)
                    video_lengths = feats_lengths.clone()
                    feats = feats[:,:max(feats_lengths)]
                    video = video[:,:max(feats_lengths)]
        
        # print("-------------- before encoder---------------",file=log_file)
        # print(f"video:{video.shape} length:{video_lengths},feat:{feats.shape},length:{feats_lengths},channels:{channels_feats.shape},length:{channels_feats_lengths}",file=log_file)
       
        # 4. Forward encoder
        # feats: B,T,D->B,T,D
        encoder_out, encoder_out_lens, _ = self.encoder(feats,feats_lengths,video,video_lengths,channels_feats) #channel_num,B,T,80 
        # print("--------------after encoder---------------",file=log_file)
        # print(f"hidden_shape:{encoder_out.shape} length:{encoder_out_lens}",file=log_file)
        
        # Post-encoder, e.g. NLU
        if self.postencoder is not None:
            encoder_out, encoder_out_lens = self.postencoder(
                encoder_out, encoder_out_lens
            )

        assert encoder_out.size(0) == speech.size(0), (
            encoder_out.size(),
            speech.size(0),
        )
        assert encoder_out.size(1) <= encoder_out_lens.max(), (
            encoder_out.size(),
            encoder_out_lens.max(),
        )

        return encoder_out, encoder_out_lens


    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
        video: torch.Tensor,
        video_lengths: torch.Tensor,
        channels: torch.Tensor,
        channels_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
       
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
            == video.shape[0]
            == video_lengths.shape[0]
            == channels.shape[0]
            == channels_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape,video.shape,text_lengths.shape,channels.shape,channels_lengths.shape)
    

        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths,video,video_lengths,channels,channels_lengths)
    
        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        stats = dict()
    
        # 1. CTC branch
        if self.ctc_weight != 0.0:
            loss_ctc, cer_ctc = self._calc_ctc_loss(
                encoder_out, encoder_out_lens, text, text_lengths
            )
                # Collect CTC branch stats
            stats["loss_ctc"] = loss_ctc.detach() if loss_ctc is not None else None
            stats["cer_ctc"] = cer_ctc

        if self.use_transducer_decoder:
            # 2a. Transducer decoder branch
            (
                loss_transducer,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_loss(
                encoder_out,
                encoder_out_lens,
                text,
            )

            if loss_ctc is not None:
                loss = loss_transducer + (self.ctc_weight * loss_ctc)
            else:
                loss = loss_transducer

            # Collect Transducer branch stats
            stats["loss_transducer"] = (
                loss_transducer.detach() if loss_transducer is not None else None
            )
            stats["cer_transducer"] = cer_transducer
            stats["wer_transducer"] = wer_transducer

        else:
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

            # Collect Attn branch stats
            stats["loss_att"] = loss_att.detach() if loss_att is not None else None
            stats["acc"] = acc_att
            stats["cer"] = cer_att
            stats["wer"] = wer_att
            
            

        # Collect total loss stats
        stats["loss"] = loss.detach()

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight




