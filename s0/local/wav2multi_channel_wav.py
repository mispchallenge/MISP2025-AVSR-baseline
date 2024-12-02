import os
import soundfile as sf
import argparse
import numpy as np
from tqdm import tqdm
from scipy.io import wavfile

def time_to_seconds(time_str):
    parts = time_str.split(':')
    parts.reverse()
    # import pdb;pdb.set_trace()
    total_seconds = 0
    if len(parts) > 0:
        s, ms = parts[0].split('.')
        total_seconds += int(s) + int(ms) / 1000.0
    if len(parts) > 1:
        total_seconds += int(parts[1]) * 60
    if len(parts) > 2:
        total_seconds += int(parts[2]) * 3600
    return total_seconds

def process_directory(root_dir, output_dir):
    for subdir, dirs, files in tqdm(os.walk(root_dir), desc='Processing directories'):
        if 'timestamp.txt' in files and any(fname.endswith('.pcm') for fname in files):

            pcm_filename = os.path.join(subdir, next(fname for fname in files if fname.endswith('.pcm')))
            timestamps_filename = os.path.join(subdir, 'timestamp.txt')
            wav_filename = os.path.join(subdir, next(fname for fname in files if fname.endswith('.wav')))
            # import pdb;pdb.set_trace()

            samplerate, data= wavfile.read(wav_filename)
            with open(timestamps_filename, 'r') as f:
                lines = f.readlines()
                start_time = time_to_seconds(lines[0].strip())
                # 如果只有开始时间，结束时间默认为最后
                # import pdb;pdb.set_trace()
                if len(lines) > 1:
                    end_time = time_to_seconds(lines[1].strip())
                else:
                    end_time = len(data) / samplerate  
            # import pdb;pdb.set_trace()


            start_sample = int(start_time * samplerate)
            end_sample = int(end_time * samplerate)

            segment_data = data[start_sample:end_sample, :]

            os.makedirs(output_dir, exist_ok=True)
            
            # 保存单通道音频文件
            for ch in range(segment_data.shape[1]):
                output_filename = os.path.join(output_dir, f'{os.path.basename(subdir)}_{ch}.wav')
                wavfile.write(output_filename , samplerate, segment_data[:, ch])
                # sf.write(output_filename, segment_data[:, ch], samplerate)
            
            with open(os.path.join(output_dir, 'segments_paths.txt'), 'a') as txt_file:
                for ch in range(segment_data.shape[1]):
                    txt_file.write(os.path.join(output_dir, f'{os.path.basename(subdir)}_{ch}.wav') + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('root_dir', type=str, default='', help='directory of single channel wav dir')
    parser.add_argument('output_dir', type=str, default='', help='directory of multi channel wav dir')
    args = parser.parse_args()

    print("Preparing multi channel wav files of {}".format(args.output_dir))
    process_directory(args.root_dir, args.output_dir)