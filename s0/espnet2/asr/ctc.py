import logging

import torch
import torch.nn.functional as F
from typeguard import check_argument_types
import numpy as np 

class CTC(torch.nn.Module):
    """CTC module.

    Args:
        odim: dimension of outputs
        encoder_output_sizse: number of encoder projection units
        dropout_rate: dropout rate (0.0 ~ 1.0)
        ctc_type: builtin or warpctc
        reduce: reduce the CTC loss into a scalar
    """

    def __init__(
        self,
        odim: int,
        encoder_output_sizse: int,
        dropout_rate: float = 0.0,
        ctc_type: str = "builtin",
        reduce: bool = True,
        ignore_nan_grad: bool = True,
    ):
        assert check_argument_types()
        super().__init__()
        eprojs = encoder_output_sizse
        self.dropout_rate = dropout_rate
        self.ctc_lo = torch.nn.Linear(eprojs, odim)
        self.ctc_type = ctc_type
        self.ignore_nan_grad = ignore_nan_grad

        if self.ctc_type == "builtin":
            self.ctc_loss = torch.nn.CTCLoss(reduction="none")
        elif self.ctc_type == "warpctc":
            import warpctc_pytorch as warp_ctc

            if ignore_nan_grad:
                logging.warning("ignore_nan_grad option is not supported for warp_ctc")
            self.ctc_loss = warp_ctc.CTCLoss(size_average=True, reduce=reduce)

        elif self.ctc_type == "gtnctc":
            from espnet.nets.pytorch_backend.gtn_ctc import GTNCTCLossFunction

            self.ctc_loss = GTNCTCLossFunction.apply
        else:
            raise ValueError(
                f'ctc_type must be "builtin" or "warpctc": {self.ctc_type}'
            )

        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen,mixinfo=None) -> torch.Tensor:
        if self.ctc_type == "builtin":
            th_pred = th_pred.log_softmax(2)
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            # logging.info(f"loss.requires_grad:{loss.requires_grad}")
            if loss.requires_grad and self.ignore_nan_grad:
                # ctc_grad: (L, B, O)
                ctc_grad = loss.grad_fn(torch.ones_like(loss))
                ctc_grad = ctc_grad.sum([0, 2])
                indices = torch.isfinite(ctc_grad)
                size = indices.long().sum()
                if size == 0:
                    # Return as is
                    logging.warning(
                        "All samples in this mini-batch got nan grad."
                        " Returning nan value instead of CTC loss"
                    )
                elif size != th_pred.size(1):
                    logging.warning(
                        f"{th_pred.size(1) - size}/{th_pred.size(1)}"
                        " samples got nan grad."
                        " These were ignored for CTC loss."
                    )

                    # Create mask for target
                    target_mask = torch.full(
                        [th_target.size(0)],
                        1,
                        dtype=torch.bool,
                        device=th_target.device,
                    )
                    s = 0
                    for ind, le in enumerate(th_olen):
                        if not indices[ind]:
                            target_mask[s : s + le] = 0
                        s += le

                    # Calc loss again using maksed data
                    loss = self.ctc_loss(
                        th_pred[:, indices, :],
                        th_target[target_mask],
                        th_ilen[indices],
                        th_olen[indices],
                    )
            else:
                size = th_pred.size(1)
                indices = torch.tensor([True]*loss.shape[0])
            ## we only set mixup for buildin CTC
            if mixinfo == None:
                if self.reduce:
                    # Batch-size average
                    loss = loss.sum() / size          
                else:
                    loss = loss / size
            else:
                idexes,alpha,bsize = mixinfo["idexes"], mixinfo["alpha"], mixinfo["bsize"]
                new_idexes,new_bsize = get_newidexes(indices.cpu().data.numpy(),idexes,bsize)
                alpha = torch.tensor(alpha).to(loss.device) 
                new_bsize = torch.tensor(new_bsize).to(loss.device)
                if self.reduce:  
                    
                    loss = get_sum(loss,new_idexes,alpha) / new_bsize
                else:
                    loss = get_sum2(loss,new_idexes,alpha) / new_bsize
            return loss

        elif self.ctc_type == "warpctc":
            # warpctc only supports float32
            th_pred = th_pred.to(dtype=torch.float32)

            th_target = th_target.cpu().int()
            th_ilen = th_ilen.cpu().int()
            th_olen = th_olen.cpu().int()
            loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            if self.reduce:
                # NOTE: sum() is needed to keep consistency since warpctc
                # return as tensor w/ shape (1,)
                # but builtin return as tensor w/o shape (scalar).
                loss = loss.sum()
            return loss

        elif self.ctc_type == "gtnctc":
            log_probs = torch.nn.functional.log_softmax(th_pred, dim=2)
            return self.ctc_loss(log_probs, th_target, th_ilen, 0, "none")

        else:
            raise NotImplementedError

    def forward(self, hs_pad, hlens, ys_pad, ys_lens,mixinfo=None):
        """Calculate CTC loss.

        Args:
            hs_pad: batch of padded hidden state sequences (B, Tmax, D)
            hlens: batch of lengths of hidden state sequences (B)
            ys_pad: batch of padded character id sequence tensor (B, Lmax)
            ys_lens: batch of lengths of character sequence (B)
        """
        # hs_pad: (B, L, NProj) -> ys_hat: (B, L, Nvocab)
        ys_hat = self.ctc_lo(F.dropout(hs_pad, p=self.dropout_rate))

        if self.ctc_type == "gtnctc":
            # gtn expects list form for ys
            ys_true = [y[y != -1] for y in ys_pad]  # parse padded ys
        else:
            # ys_hat: (B, L, D) -> (L, B, D)
            ys_hat = ys_hat.transpose(0, 1)
            # (B, L) -> (BxL,)
            ys_true = torch.cat([ys_pad[i, :l] for i, l in enumerate(ys_lens)])
    
        loss = self.loss_fn(ys_hat, ys_true,hlens,ys_lens,mixinfo).to(
            device=hs_pad.device, dtype=hs_pad.dtype
        )

        return loss

    def log_softmax(self, hs_pad):
        """log_softmax of frame activations

        Args:
            Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: log softmax applied 3d tensor (B, Tmax, odim)
        """
        return F.log_softmax(self.ctc_lo(hs_pad), dim=2)

    def argmax(self, hs_pad):
        """argmax of frame activations

        Args:
            torch.Tensor hs_pad: 3d tensor (B, Tmax, eprojs)
        Returns:
            torch.Tensor: argmax applied 2d tensor (B, Tmax)
        """
        return torch.argmax(self.ctc_lo(hs_pad), dim=2)


def check(idexes,bsize):#[0,0,0,1,-1,1,-1]->[0,0,0,-1,-1,-1,-1]
    # assert idexes[:bsize].all() == 0 and idexes[bsize:].all() != 0
    for i in range(bsize,len(idexes),2):
        # import pdb;pdb.set_trace()
        if -1 in idexes[[i,i+1]]:
            idexes[i] = -1 
            idexes[i+1] = -1
    return idexes

# [0,0,0,1,2,1,2] [False,True,True,True,False,True,True] bsize=3->[-1,0,0,-1,2,1,2]->[-1,0,0,-1,-1,1,2]->[0,0,-1,1,2]
# bsize = 3
# idexes = np.array([0,0,0,1,2,1,2])
# mask = np.array([False,True,True,True,False,True,True])
def get_newidexes(mask,idexes,bsize):
    remask = (np.negative(mask.astype(int))+1).astype(bool)
    idexes[remask] = -1 
    idexes = check(idexes,bsize)
    new_idexes = idexes[mask]
    zero_count = np.where(new_idexes==0,1,0).sum()
    negative_count = np.where(new_idexes==-1,1,0).sum()
    bsize = int((len(new_idexes) + zero_count - negative_count)/2)
    return new_idexes,bsize

def get_sum(loss,index,alpha):
    return loss[np.concatenate(np.argwhere(index==0))].sum() + loss[np.concatenate(np.argwhere(index==1))].sum()*alpha + loss[np.concatenate(np.argwhere(index==2))].sum()*(1-alpha)

def get_sum2(loss,index,alpha):
    newsamples = []
    zero_index = np.concatenate(np.argwhere(index==0))
    one_index = np.concatenate(np.argwhere(index==1))
    two_index = np.concatenate(np.argwhere(index==2))
    assert len(one_index),len(two_index)
    for one,two in zip(one_index,two_index):
        newsamples.append(loss[one]*alpha + two[one]*(1-alpha))
        
    return torch.cat((loss[zero_index],newsamples),0)