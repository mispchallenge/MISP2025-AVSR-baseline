import os
import subprocess

source_dir = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/misp_data/eval-part_wave'
target_dir = '/train33/sppro/permanent/hangchen2/gss/gss_pro/gss_main_code/gss_main/gss_main/misp_data/eval-part_wave_v6'

for root, dirs, files in os.walk(source_dir):
    for file in files:
        if file.endswith('.flac'):
            rel_dir = os.path.relpath(root, source_dir)
            source_file = os.path.join(root, file)
            target_file_dir = os.path.join(target_dir, rel_dir)
            target_file = os.path.join(target_file_dir, file.replace('.flac', '.wav'))
            os.makedirs(target_file_dir, exist_ok=True)
            subprocess.run(['ffmpeg', '-i', source_file, target_file])

print("Done.")