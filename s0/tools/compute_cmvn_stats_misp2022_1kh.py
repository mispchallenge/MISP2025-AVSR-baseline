#!/usr/bin/env python3
# encoding: utf-8

import sys
import argparse
import json
import codecs
import yaml

import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
from torch.utils.data import Dataset, DataLoader

torchaudio.set_audio_backend("sox_io")


class CollateFunc(object):
    ''' Collate function for AudioDataset
    '''

    def __init__(self, feat_dim, resample_rate):
        self.feat_dim = feat_dim
        self.resample_rate = resample_rate
        pass

    def __call__(self, batch):
        mean_stat = torch.zeros(self.feat_dim)
        var_stat = torch.zeros(self.feat_dim)
        number = 0
        for item in batch:
            # print(item) 元组
            value = item[1].strip().split(",")
            # print(value) list型
            assert len(value) == 3 or len(value) == 1
            wav_path = value[0]
            sample_rate = 16000
            resample_rate = sample_rate
            # len(value) == 3 means segmented wav.scp,
            # len(value) == 1 means original wav.scp
            # 对于拼接数据需要用不同的处理方式!
            if len(value) == 3:
                start_frame = int(float(value[1]) * sample_rate)
                end_frame = int(float(value[2]) * sample_rate)
                waveform, sample_rate = torchaudio.backend.sox_io_backend.load(
                    filepath=wav_path,
                    num_frames=end_frame - start_frame,
                    frame_offset=start_frame)
            else:
                index = wav_path.find('@')
                sample_rate = 16000
                if index == -1:
                    # 非拼接音频
                    # 发现有的音频读入后shape为[X, 1], 需要reshape
                    # /yrfs2/cv1/hangchen2/espnet/espnet-master/egs2/mispi/avsr/dump/raw/org/train_near_sp/data/format.1/S008_R01_S006007008009_C07_I1_098792-099096.pt
                    # tensor([[  1],
                    #         [  4],
                    #         [  2],
                    #         ...,
                    #         [-17],
                    #         [-18],
                    #         [-19]], dtype=torch.int16)
                    waveform = torch.load(item[1])
                    if len(waveform.shape) != 1:
                        length = waveform.shape[0]
                        waveform = waveform.reshape(length)
                else:
                    # 拼接音频
                    wav_path_1 = wav_path[0:index]      # print(wav_path_1)
                    wav_path_2 = wav_path[index+1:]     # print(wav_path_2)
                    waveform_1 = torch.load(wav_path_1)
                    if len(waveform_1.shape) != 1:
                        length_1 = waveform_1.shape[0]
                        waveform_1 = waveform_1.reshape(length_1)   # print(waveform_1)
                    waveform_2 = torch.load(wav_path_2)
                    if len(waveform_2.shape) != 1:
                        length_2 = waveform_2.shape[0]
                        waveform_2 = waveform_2.reshape(length_2)   # print(waveform_2)
                    waveform = torch.cat((waveform_1, waveform_2))
            
            # print(waveform)
            waveform = waveform * (1 << 15)
            if self.resample_rate != 0 and self.resample_rate != sample_rate:
                resample_rate = self.resample_rate
                waveform = torchaudio.transforms.Resample(
                    orig_freq=sample_rate, new_freq=resample_rate)(waveform)

            waveform = waveform.unsqueeze(0).type(torch.float32)
            # print(waveform)
            # waveform: <class 'torch.Tensor'>
            #           torch.float32
            #           torch.Size([1, 56767])
            mat = kaldi.fbank(waveform,
                              num_mel_bins=self.feat_dim,
                              dither=0.0,
                              energy_floor=0.0,
                              sample_frequency=resample_rate)
            mean_stat += torch.sum(mat, axis=0)
            var_stat += torch.sum(torch.square(mat), axis=0)
            number += mat.shape[0]
        return number, mean_stat, var_stat


class AudioDataset(Dataset):
    def __init__(self, data_file):
        self.items = []
        with codecs.open(data_file, 'r', encoding='utf-8') as f:
            for line in f:
                arr = line.strip().split()
                self.items.append((arr[0], arr[1]))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='extract CMVN stats')
    parser.add_argument('--num_workers',
                        default=0,
                        type=int,
                        help='num of subprocess workers for processing')
    parser.add_argument('--train_config',
                        default='',
                        help='training yaml conf')
    parser.add_argument('--in_scp', default=None, help='wav scp file')
    parser.add_argument('--out_cmvn',
                        default='global_cmvn',
                        help='global cmvn file')

    doc = "Print log after every log_interval audios are processed."
    parser.add_argument("--log_interval", type=int, default=1000, help=doc)
    args = parser.parse_args()

    with open(args.train_config, 'r') as fin:
        configs = yaml.load(fin, Loader=yaml.FullLoader)
    feat_dim = configs['dataset_conf']['fbank_conf']['num_mel_bins']
    resample_rate = 0
    if 'resample_conf' in configs['dataset_conf']:
        resample_rate = configs['dataset_conf']['resample_conf']['resample_rate']
        print('using resample and new sample rate is {}'.format(resample_rate))

    collate_func = CollateFunc(feat_dim, resample_rate)
    dataset = AudioDataset(args.in_scp)
    batch_size = 20
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True,
                             sampler=None,
                             num_workers=args.num_workers,
                             collate_fn=collate_func)

    with torch.no_grad():
        all_number = 0
        all_mean_stat = torch.zeros(feat_dim)
        all_var_stat = torch.zeros(feat_dim)
        wav_number = 0
        for i, batch in enumerate(data_loader):
            number, mean_stat, var_stat = batch
            all_mean_stat += mean_stat
            all_var_stat += var_stat
            all_number += number
            wav_number += batch_size

            if wav_number % args.log_interval == 0:
                print(f'processed {wav_number} wavs, {all_number} frames',
                      file=sys.stderr,
                      flush=True)

    cmvn_info = {
        'mean_stat': list(all_mean_stat.tolist()),
        'var_stat': list(all_var_stat.tolist()),
        'frame_num': all_number
    }

    with open(args.out_cmvn, 'w') as fout:
        fout.write(json.dumps(cmvn_info))
