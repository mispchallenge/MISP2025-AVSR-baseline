import os
from pydub import AudioSegment
from moviepy.editor import VideoFileClip, concatenate_videoclips
import argparse

def merge_audio_files(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for scene_folder in os.listdir(source_dir):
        scene_path = os.path.join(source_dir, scene_folder)

        if os.path.isdir(scene_path):
            for psc_folder in os.listdir(scene_path):
                psc_path = os.path.join(scene_path, psc_folder)

                if os.path.isdir(psc_path):
                    wav_files = [f for f in os.listdir(psc_path) if f.endswith('.wav')]
                    wav_files.sort(key=lambda x: int(x.split('-')[-1].split('.')[0]))

                    combined = AudioSegment.empty()
                    for wav_file in wav_files:
                        wav_path = os.path.join(psc_path, wav_file)
                        audio = AudioSegment.from_wav(wav_path)
                        combined += audio
                    
                    output_filename = f"{psc_folder}.wav"
                    #print(output_filename)
                    output_path = os.path.join(target_dir, output_filename)
                    combined.export(output_path, format='wav')

                    print(f"Merge audio file:{output_path}")

def merge_video_files(source_dir, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    for scene_folder in os.listdir(source_dir):
        scene_path = os.path.join(source_dir, scene_folder)
        if scene_folder in ['B661', 'B374', 'B451', 'D229', 'S495', 'B313', \
            'S444', 'B315', 'B420', 'D253', 'B314', 'D495', 'B319', 'A230', \
            'B368', 'D224']:
            print(f"{scene_folder} already done.")
            continue
        if os.path.isdir(scene_path):
            for psc_folder in os.listdir(scene_path):
                psc_path = os.path.join(scene_path, psc_folder)

                if os.path.isdir(psc_path):
                    mp4_files = [f for f in os.listdir(psc_path) if f.endswith('.mp4')]
                    angles = ['0', '120', '240']
                    for angle in angles:
                        angle_files = [f for f in mp4_files if f.endswith(f'-{angle}.mp4')]
                        
                        angle_files.sort(key=lambda x: int(x.split('-')[-2]))

                        video_clips = []
                        for mp4_file in angle_files:
                            mp4_path = os.path.join(psc_path, mp4_file)
                            print(mp4_path)
                            try:
                                video_clips.append(VideoFileClip(mp4_path))
                            except Exception as e:
                                print(f"Wrong:{mp4_path}")
                                print(e)
                        
                        if video_clips:
                            final_clip = concatenate_videoclips(video_clips, method="compose")

                            output_filename = f"{psc_folder}-{angle}.mp4"
                            output_path = os.path.join(target_dir, output_filename)
                            final_clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                            print(f"Merge audio file:{output_path}")



if __name__ == "__main__":
    parser = argparse.ArgumentParser('')
    parser.add_argument('in_video_path', type=str, default='', help='input video path')
    parser.add_argument('out_video_path', type=str, default='', help='output video path')
    args = parser.parse_args()

    merge_video_files(args.in_video_path, args.out_video_path)
