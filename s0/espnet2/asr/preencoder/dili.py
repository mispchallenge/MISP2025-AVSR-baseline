import torch 
from  espnet2.asr.preencoder.wav import GatecnnfeatPreEncoder
frontend = GatecnnfeatPreEncoder()
maxlen = 100
feats = torch.rand(16,maxlen,80) 
lengths = torch.randint(maxlen-1,maxlen,(16,))
output,output_length = frontend(feats,lengths)#[B,T,D]->[B,T,D]
print(output.shape,output_length)
# if max_seq_len%64 != 0:
#             src_audio = torch.nn.functional.pad(src_audio, pad=(0, 0, 0, 64 - max_seq_len%64), mode='constant', value=0)
#             src_lengths_audio = src_lengths_audio + (64 - max_seq_len%64)
#             _, max_seq_len, _ = src_audio.size()