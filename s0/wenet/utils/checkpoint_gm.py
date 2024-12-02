# Copyright (c) 2023 (Zhe Wang)
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

import logging
import os
import re

import yaml
import torch
from collections import OrderedDict

import datetime


def load_checkpoint(model: torch.nn.Module, path: str) -> dict:
    if torch.cuda.is_available():
        logging.info('Checkpoint: loading from checkpoint %s for GPU' % path)
        checkpoint = torch.load(path)
    else:
        logging.info('Checkpoint: loading from checkpoint %s for CPU' % path)
        checkpoint = torch.load(path, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    info_path = re.sub('.pt$', '.yaml', path)
    configs = {}
    if os.path.exists(info_path):
        with open(info_path, 'r') as fin:
            configs = yaml.load(fin, Loader=yaml.FullLoader)
    return configs


def save_checkpoint(model: torch.nn.Module, path: str, infos=None):
    '''
    Args:
        infos (dict or None): any info you want to save.
    '''
    logging.info('Checkpoint: save to checkpoint %s' % path)
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    elif isinstance(model, torch.nn.parallel.DistributedDataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    torch.save(state_dict, path)
    info_path = re.sub('.pt$', '.yaml', path)
    if infos is None:
        infos = {}
    infos['save_time'] = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')
    with open(info_path, 'w') as fout:
        data = yaml.dump(infos)
        fout.write(data)


def filter_modules(model_state_dict, modules):
    new_mods = []
    incorrect_mods = []
    mods_model = model_state_dict.keys()
    # print(mods_model)
    for mod in modules:
        if any(key.startswith(mod) for key in mods_model):
            new_mods += [mod]  # mod :encoder.encoders
        else:
            incorrect_mods += [mod]
    if incorrect_mods:
        logging.warning(
            "module(s) %s don't match or (partially match) "
            "available modules in model.",
            incorrect_mods,
        )
        logging.warning("for information, the existing modules in model are:")
        logging.warning("%s", mods_model)

    return new_mods


def load_trained_modules(model: torch.nn.Module, args: None):
    # Load encoder modules with pre-trained model(s).
    enc_model_path = args.enc_init
    enc_modules = args.enc_init_mods
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    if os.path.isfile(enc_model_path):
        logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                     enc_model_path)
        model_state_dict = torch.load(enc_model_path, map_location='cpu')
        modules = filter_modules(model_state_dict, enc_modules)
        partial_state_dict = OrderedDict()
        for key, value in model_state_dict.items():
            if any(key.startswith(m) for m in modules):
                partial_state_dict[key] = value
        main_state_dict.update(partial_state_dict)
    else:
        logging.warning("model was not found : %s", enc_model_path)

    model.load_state_dict(main_state_dict)
    configs = {}
    return configs

def change_key(key, enc_modules):
    new_key_vec = []
    key_vec = key.split(".")
    for src_key, tg_key in enc_modules.items(): 
        src_key_vec = src_key.split(".")
        tg_key_vec = tg_key.split(".")
        flag = 1
        if len(src_key_vec) <= len(key_vec):
            for i in range(len(src_key_vec)):
                if key_vec[i] != src_key_vec[i]:
                    flag = 0
        if flag == 1:
            new_key_vec += tg_key_vec
            new_key_vec += key_vec[len(src_key_vec):]
    new_key = ".".join(new_key_vec)
    # print(new_key)
    return new_key

# zw
def load_pretrain_modules(model: torch.nn.Module, init_param: dict, name_map: dict, device):
    main_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    for i in range(len(init_param)):
        enc_model_path = init_param[i]   
        enc_modules = name_map[i]
        if os.path.isfile(enc_model_path):
            if device == torch.device('cuda'):
                logging.info('Checkpoint: loading from checkpoint %s for GPU' %
                        enc_model_path)
            elif device == torch.device('cpu'):
                logging.info('Checkpoint: loading from checkpoint %s for CPU' %
                        enc_model_path)
            # model_state_dict = torch.load(enc_model_path, map_location='cpu')
            model_state_dict = torch.load(enc_model_path, map_location=device)
            # print(enc_modules) # {'encoder.encoders': 'encoder.cross_fusion_encoderlayers'}
            modules = filter_modules(model_state_dict, enc_modules)
            # print(modules)  # ['encoder.encoders']   ['video_frontend.resnet', 'video_frontend.video_frontend']
            partial_state_dict = OrderedDict()
            for key, value in model_state_dict.items():
                if any(key.startswith(m) for m in modules):
                    new_key = change_key(key, enc_modules)
                    partial_state_dict[new_key] = value
            # print(partial_state_dict)
            main_state_dict.update(partial_state_dict)
            # print(main_state_dict)
        else:
            logging.warning("model was not found : %s", enc_model_path)
    # import pdb; pdb.set_trace()
    model.load_state_dict(main_state_dict)
    configs = {}
    return configs

# gm
def load_visual_pretrain_modules(model: torch.nn.Module, init_param: dict, name_map: dict, device):
    current_state_dict = model.state_dict()
    logging.warning("model(s) found for pre-initialization")
    uninit_params = set(current_state_dict.keys())
    init_params = set()
    for pretrained_path, nmap in zip(init_param, name_map):
        if os.path.isfile(pretrained_path):
            if device == torch.device('cuda'):
                logging.info('Checkpoint: loading from checkpoint %s for GPU' % pretrained_path)
            elif device == torch.device('cpu'):
                logging.info('Checkpoint: loading from checkpoint %s for CPU' % pretrained_path)
            pretrained_state_dict = torch.load(pretrained_path, map_location=device)
            if 'model_state_dict' in pretrained_state_dict:
                pretrained_state_dict = pretrained_state_dict['model_state_dict']
            mapped_state_dict = {}

            for current_name in current_state_dict.keys():
                mapped = False
                for prefix, mapped_prefix in nmap.items():
                    if current_name.startswith(prefix):
                        pretrained_name = current_name.replace(prefix, mapped_prefix, 1)
                        if pretrained_name in pretrained_state_dict:
                            mapped_state_dict[current_name] = pretrained_state_dict[pretrained_name]
                            mapped = True
                            #print(f'{current_name} mapped by {os.path.basename(pretrained_path)}')
                            uninit_params.discard(current_name)
                            break
                if not mapped and current_name in pretrained_state_dict:
                    mapped_state_dict[current_name] = pretrained_state_dict[current_name]
                    mapped = True
                    uninit_params.discard(current_name)
                    #print(f'{current_name} nature mapped by {os.path.basename(pretrained_path)}')
                # elif not mapped:
                #     print(f'{current_name} not mapped')

            
            current_state_dict.update(mapped_state_dict)
    
    model.load_state_dict(current_state_dict, strict=False)
    configs = {}
    return configs