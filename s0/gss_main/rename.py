
from pathlib import Path 
import argparse
from tqdm import tqdm
import numpy as np
import os
import glob
import codecs
import argparse
from multiprocessing import Pool
import sys


# def text2lines(textpath, lines_content=None):
#     """
#     read lines from text or write lines to txt
#     :param textpath: filepath of text
#     :param lines_content: list of lines or None, None means read
#     :return: processed lines content for read while None for write
#     """
#     if lines_content is None:

#     else:
#         processed_lines = list(map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content))
#         with codecs.open(textpath, 'w') as handle:
#             handle.write(''.join(processed_lines))
#         return None

#  python /train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/rename.py --input "dump/raw/gssgpu_train_far_flac/speech_shape_raw" --output "dump/rawgssgpu_train_far_flac/speech_shape_v1
def rename_audio_file(inputfile, lines_content=None):
    if lines_content is None:
        with codecs.open(inputfile, 'r') as handle:
            lines_content = handle.readlines()
        processed_lines = list(map(lambda x: x[:-1] if x[-1] in ['\n'] else x, lines_content))
    
        for c_k in range(len(processed_lines)):
            rename_part =  processed_lines[c_k].split(' ')[0]
            replaced_string = rename_part .replace("-", "_")
            split_chunks = replaced_string .split("_")
            new_name  = split_chunks[4]+'_'+'_'.join(split_chunks[0:3])+'_'+'_'.join(split_chunks[5:7])
            processed_lines[c_k] = new_name + ' ' + processed_lines[c_k].split(' ')[1]
            # import pdb;pdb.set_trace()
        return processed_lines
    else:
        processed_lines = list(map(lambda x: x if x[-1] in ['\n'] else '{}\n'.format(x), lines_content))
        with codecs.open(inputfile, 'w') as handle:
            handle.write(''.join(processed_lines))
        return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('--input', type=str, default='')
    args = parser.parse_args()
    input = Path(args.input)
    processed_lines = rename_audio_file(input,None)
    lines = sorted(processed_lines)
    # import pdb;pdb.set_trace()
    rename_audio_file(input,lines)