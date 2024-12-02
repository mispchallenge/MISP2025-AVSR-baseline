#!/usr/bin/env python3

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

import os
import glob
import argparse
import codecs
import json
from tqdm import tqdm
from multiprocessing import Pool


from tool_misp2022.data_io import safe_load, safe_store

def prepare_wav_scp(audio_scp, video_scp, av_scp):
    a_table = {}
    with open(audio_scp, 'r', encoding='utf8') as fa:
        for line in fa:
            arr = line.strip().split()
            assert len(arr) == 2
            a_table[arr[0]] = arr[1]
    v_table = {}
    with open(video_scp, 'r', encoding='utf8') as fv:
        for line in fv:
            arr = line.strip().split()
            assert len(arr) == 2
            v_table[arr[0]] = arr[1]
    
    av_lines = []
    for key, a_path in a_table.items():
        if key in v_table.keys():
            av_lines.append('{} {} {}'.format(key, a_path, v_table[key]))
    safe_store(file=av_scp, data=sorted(av_lines), mode='cover', ftype='txt')
    return None

def prepare_text(audio_text, video_text, av_text):
    a_table = {}
    with open(audio_text, 'r', encoding='utf8') as fa:
        for line in fa:
            arr = line.strip().split()
            assert len(arr) == 2
            a_table[arr[0]] = arr[1]
    v_table = {}
    with open(video_text, 'r', encoding='utf8') as fv:
        for line in fv:
            arr = line.strip().split()
            assert len(arr) == 2
            v_table[arr[0]] = arr[1]
    
    av_lines = []
    for key, a_path in a_table.items():
        if key in v_table.keys():
            av_lines.append('{} {}'.format(key, a_path))
    safe_store(file=av_text, data=sorted(av_lines), mode='cover', ftype='txt')
    return None

def prepare_data_list(av_scp, av_text, av_data_list):
    av_text_table = {}
    with open(av_text, 'r', encoding='utf8') as fav_text:
        for line in fav_text:
            arr = line.strip().split()
            assert len(arr) == 2
            av_text_table[arr[0]] = arr[1]
    with open(av_scp, 'r', encoding='utf8') as fav_scp, \
         open(av_data_list, 'w', encoding='utf8') as fav_data_list:
        for line in fav_scp:
            arr = line.strip().split()
            assert len(arr) == 3
            key = arr[0]
            wav = arr[1]
            video = arr[2]
            if key in av_text_table:
                txt = av_text_table[key]
                line = dict(key=key, wav=wav, video=video, txt=txt)
            else:
                # 由于数据标注是根据近场标注，远场视频可能断。所以没有对应的.pt文件。此处直接continue
                continue
            json_line = json.dumps(line, ensure_ascii=False)
            fav_data_list.write(json_line + '\n')
    return None

def prepare_wenet_av_data_files(audio_dir, video_dir, av_dir):
    a_scp, a_text = os.path.join(audio_dir, 'wav.scp'), os.path.join(audio_dir, 'text')
    v_scp, v_text = os.path.join(video_dir, 'wav.scp'), os.path.join(video_dir, 'text')
    av_scp, av_text, av_data_list= os.path.join(av_dir, 'wav.scp'), os.path.join(av_dir, 'text'), os.path.join(av_dir, 'data.list')
    prepare_wav_scp(a_scp, v_scp, av_scp)
    prepare_text(a_text, v_text, av_text)
    prepare_data_list(av_scp, av_text, av_data_list)
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('audio_dir', type=str, default='', help='directory of audio data')
    parser.add_argument('video_dir', type=str, default='', help='directory of video data')
    parser.add_argument('av_dir', type=str, default='', help='directory of audio-visual data')
    args = parser.parse_args()

    print('Preparing data.list/text with {} and {}'.format(args.audio_dir, args.video_dir))
    prepare_wenet_av_data_files(audio_dir=args.audio_dir, video_dir=args.video_dir, av_dir=args.av_dir)