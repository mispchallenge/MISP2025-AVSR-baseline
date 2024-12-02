import os
import shutil
import argparse
from tqdm import tqdm

# copy textgrid to a new file
def find_and_copy_textgrids(src_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    
    # 遍历源目录及其子目录
    for subdir, _, files in tqdm(os.walk(src_dir), desc='Scanning directories'):
        for file in files:
            # import pdb;pdb.set_trace()
            if file.endswith('-cutoff.TextGrid'):
                ## gm-added start
                dest_file = file.replace("-cutoff", "")
                ## gm-added end
                # 构建源文件路径
                src_file_path = os.path.join(subdir, file)
                # 构建目标文件路径
                dest_file_path = os.path.join(dest_dir, dest_file) ## gm-changed
                # 复制文件
                shutil.copy2(src_file_path, dest_file_path)
                print(f"Copied {src_file_path} to {dest_file_path}")


def find_and_rename_files(wav_dir, transcript_dir, dest_wav_dir, dest_textgrid_dir):
    os.makedirs(dest_wav_dir, exist_ok=True)
    os.makedirs(dest_textgrid_dir, exist_ok=True)

    # 找到所有.TextGrid文件
    textgrid_files = [f for f in os.listdir(transcript_dir) if f.endswith('.TextGrid')]
    # 创建一个字典，以前四个字符为键，存储相关的TextGrid文件名
    textgrid_dict = {}
    for tg_file in textgrid_files:
        prefix = tg_file[:4]
        if prefix not in textgrid_dict:
            textgrid_dict[prefix] = []
        textgrid_dict[prefix].append(tg_file)

    # 找到所有.wav文件
    wav_files = [f for f in os.listdir(wav_dir) if f.endswith('.wav')]

    for wav_file in tqdm(wav_files, desc='Processing .wav files'):
        # 提取音频文件的前缀名
        audio_prefix = wav_file[:4]

        # 如果有匹配的TextGrid文件
        if audio_prefix in textgrid_dict:
            # 生成新的文件名
            
            new_file_suffixes = sorted([tg.split('-')[-1].split('.')[0] for tg in textgrid_dict[audio_prefix]], key=lambda x: int(x))
            new_file_suffix = ''.join(new_file_suffixes)
            new_file_name = f"{audio_prefix}_S{new_file_suffix}_F8N_Far_{wav_file.split('_')[-1][0]}.wav"
            # 构建源文件路径和目标文件路径
            src_file_path = os.path.join(wav_dir, wav_file)
            dest_file_path = os.path.join(dest_wav_dir, new_file_name)
            # import pdb;pdb.set_trace()
            # 复制并重命名文件
            shutil.copy2(src_file_path, dest_file_path)
            print(f"Copied and renamed {src_file_path} to {dest_file_path}")

            # 处理TextGrid文件
            for tg_file in textgrid_dict[audio_prefix]:
                tg_suffix = tg_file.split('-')[-1].split('.')[0]
                new_tg_file_name = f"{audio_prefix}_S{new_file_suffix}_F8N_Near_{tg_suffix}.TextGrid"
                
                # 构建源文件路径和目标文件路径
                src_tg_file_path = os.path.join(transcript_dir, tg_file)
                dest_tg_file_path = os.path.join(dest_textgrid_dir, new_tg_file_name)
                
                # 复制并重命名 TextGrid 文件
                shutil.copy2(src_tg_file_path, dest_tg_file_path)
                print(f"Copied and renamed {src_tg_file_path} to {dest_tg_file_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('textgrid_src_dir', type=str, default='', help='directory of source textgrid')
    parser.add_argument('wav_src_dir', type=str, default='', help='directory of source wav')
    parser.add_argument('textgrid_raw_dir', type=str, default='', help='directory of raw textgrid')
    parser.add_argument('wav_dst_dir', type=str, default='', help='directory of destination wav')
    parser.add_argument('textgrid_dst_dir', type=str, default='', help='directory of destination textgrid')
    args = parser.parse_args()

    find_and_copy_textgrids(args.textgrid_src_dir, args.textgrid_raw_dir)
    find_and_rename_files(args.wav_src_dir, args.textgrid_raw_dir, args.wav_dst_dir, args.textgrid_dst_dir)
