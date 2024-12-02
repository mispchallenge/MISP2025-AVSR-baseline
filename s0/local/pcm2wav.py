import os
import subprocess
from tqdm import tqdm
import argparse
import codecs
import numpy as np
from scipy.io import wavfile
from datetime import datetime
# def convert_pcm_to_wav(input_pcm):
#     output_wav = os.path.splitext(input_pcm)[0] + '.wav'
#     command = [
#         'ffmpeg',
#         '-f', 's32le',
#         '-ar', '16000',
#         '-ac', '8',
#         '-i', input_pcm,
#         output_wav
#     ]
#     subprocess.run(command, check=True)

def pcm2numpy(pcm_file, channel=8, bit=32):
    with codecs.open(pcm_file, 'rb') as pcm_handle:
        pcm_frames = pcm_handle.read()
        np_array = np.frombuffer(pcm_frames, dtype='int{}'.format(bit), offset=0)
    return np_array.reshape((-1,channel))


def process_directory(directory):
    pcm_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith('.pcm'):
                pcm_files.append(os.path.join(root, file))

    for pcm_file in tqdm(pcm_files, desc="Converting PCM to WAV"):
        # import pdb;pdb.set_trace()
        data = pcm2numpy(pcm_file)
        wav_filename = pcm_file[:-4]+'.wav'
        wavfile.write(wav_filename , 16000, data)

        log_file = 'conversion_log.txt'
        with open(log_file, 'a') as f:
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            log_entry = f"{timestamp}: Converted {pcm_file} to {wav_filename}\n"
            f.write(log_entry)

if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('main_dir', type=str, default='', help='directory of pcm')
    args = parser.parse_args()

    #main_dir = '/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt/data/MISP-Meeting/training'
    print("Preparing wav files of {}".format(args.main_dir))
    process_directory(args.main_dir)