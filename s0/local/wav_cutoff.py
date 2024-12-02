import os
import argparse
from pydub import AudioSegment
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import VideoFileClip

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

def parse_timestamp_file(timestamp_file):
    """
    解析 timestamp.txt 文件，返回开始时间和结束时间（以秒为单位）。
    假设每行格式为: start_time end_time
    """
    timestamps = []
    with open(timestamp_file, 'r') as f:
        lines = f.readlines()
        if len(lines)==0:
            start_time = 0
        else:
            start_time = time_to_seconds(lines[0].strip())
        if len(lines) > 1:
            end_time = time_to_seconds(lines[1].strip())
        else:
            end_time = None
        print(start_time, end_time)
        print(timestamp_file)
        if end_time == None:
            timestamps.append((start_time, None))
        else:
            timestamps.append((start_time, end_time))

    return timestamps


def cut_mp4_file(input_mp4, timestamps, output_dir):
    """
    根据时间戳切割 wav 文件，并保存到 output_dir。
    """
    video = VideoFileClip(input_mp4)
    video_duration = video.duration

    for idx, (start_time, end_time) in enumerate(timestamps):
        # 切割音频文件
        # if end_time == None:
        #     end_time = video_duration
        end_time = video_duration
        
        # 根据重命名规则生成文件名
        filename = os.path.basename(input_mp4)
        output_file = os.path.join(output_dir, filename)
        
        # 保存文件
        ffmpeg_extract_subclip(input_mp4, start_time, end_time, targetname=output_file)
        print(f"Saved {output_file}")


def process_video_files(input_dir, output_dir, timestamp_dir):
    """
    遍历 input_dir 和 timestamp_dir，找到对应的 wav 文件和 timestamp 文件，执行切割操作。
    input_dir: .../training_video_merge       (A213-PSCx3-0.mp4)
    output_dir: .../training_video_cutoff     (A213-PSCx3-0.mp4)
    timestamp_dir: .../training/A213/A213-PSCx3/timestamp.txt
    """
    # 确保输出文件夹存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file in os.listdir(input_dir):
        scene_id = file.split('-')[0]
        input_mp4 = os.path.join(input_dir, file)
        timestamp_file = os.path.join(timestamp_dir, scene_id, scene_id+'-PSCx3', 'timestamp.txt')
        
        timestamps = parse_timestamp_file(timestamp_file)
        print(input_mp4, timestamp_file, timestamps)
        cut_mp4_file(input_mp4, timestamps, output_dir)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('video_input_dir', type=str, default='', help='directory of input video')
    parser.add_argument('video_output_dir', type=str, default='', help='directory of otuput video')
    parser.add_argument('timestamp_dir', type=str, default='', help='directory of timestamp')
    args = parser.parse_args()

    process_video_files(input_dir=args.video_input_dir, output_dir=args.video_output_dir, timestamp_dir=args.timestamp_dir)