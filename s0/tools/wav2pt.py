import os
import torch
import torchaudio

def convert_wav_to_pt(input_folder, output_folder):
    os.makedirs(output_folder, exist_ok=True)

    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.endswith(".wav"):
                input_path = os.path.join(root, filename)
                rel_path = os.path.relpath(root, input_folder)
                output_path = os.path.join(output_folder, rel_path, filename)
                #os.makedirs(os.path.dirname(output_path), exist_ok=True)
                print(filename, output_path)
                #wavform, sample_rate = torchaudio.load(input_path)
                #torch.save(wavform, output_path)
                #print(f"Converted {filename} to {os.path.basename(output_path)}")

if __name__ == "__main__":
    input_folder = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/misp_data/train_wave_v6/far/wpe/gss_new/enhanced'
    output_folder = '/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt-minggao/examples/misp2022/s0/data_far_audio/training_far_audio_segment'
    convert_wav_to_pt(input_folder, output_folder)