import os
import wave

def check_audio_files(segment_dir):
    for root, dirs, files in os.walk(segment_dir):
        for file in files:
            if file.endswith('.pt'):
                file_path = os.path.join(root, file)
                print(file_path)

                with wave.open(file_path, 'r') as wav:
                    nframes = wav.getnframes()
                    if nframes == 0:
                        print(file_path)

if __name__ == '__main__':
    segment_dir = '/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt-minggao/examples/misp2022/s0/data_far_audio/training_far_audio_segment'
    check_audio_files(segment_dir)