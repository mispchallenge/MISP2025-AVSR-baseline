import cv2, math, tqdm, os, glob
import numpy as np
from moviepy.editor import VideoFileClip
import argparse


def compute_pinhole_intrinsic_by_physic_param(fov, resolution):
    h_fov, v_fov = fov
    w, h = resolution

    fx = None
    fy = None
    if h_fov > 0:
        fx = w / (2 * math.tan((h_fov * math.pi / 180)/2))
    if v_fov > 0:
        fy = w / (2 * math.tan((v_fov * math.pi / 180)/2))
    
    if fx is None and fy is None:
        raise ValueError()
    elif fx is None:
        fx = fy
    else:
        fy = fx
    
    cx = w / 2
    cy = h / 2
    intrinsic = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    return intrinsic


def convert_angle_range(angles):
    angles = angles - 360 * np.floor((angles + 180) / 360)
    return angles


class OminiCamera:
    def __init__(self, input_resolution):
        self.input_resolution = input_resolution
        self.output_resolution = None
        self.output_center_direction = None
        self.output_fov = None
    
    def set_output_config(self, output_resolution, output_center_direction, output_fov):
        self.output_resolution = output_resolution
        self.output_center_direction = output_center_direction
        self.output_fov = output_fov
        self._compute_remap()
    
    def _compute_remap(self):
        out_w, out_h = self.output_resolution
        h_offset, v_offset = self.output_center_direction
        intrinsic = compute_pinhole_intrinsic_by_physic_param(self.output_fov, self.output_resolution)
        us, vs = np.meshgrid(np.arange(out_w), np.arange(out_h))
        zs = np.ones_like(vs)
        uvzs = np.stack([us, vs, zs], axis=-1).reshape(-1, 3).astype(np.float32)
        inv_intrinsic = np.linalg.inv(intrinsic)
        pts = np.matmul(inv_intrinsic, uvzs.T)

        h_angles = np.arctan2(pts[0], pts[2]) * 180 / np.pi
        v_angles = np.arctan2(pts[1], np.sqrt(np.power(pts[2], 2) + np.power(pts[0], 2))) * 180 / np.pi

        raw_h_angles = h_angles + h_offset
        raw_v_angles = v_angles + v_offset

        raw_h_angles = convert_angle_range(raw_h_angles)
        raw_v_angles = convert_angle_range(raw_v_angles)

        # import pdb; pdb.set_trace()

        in_w, in_h = self.input_resolution
        xs = (raw_h_angles + 180) / 360 * (in_w - 1)
        ys = (raw_v_angles + 90) / 180 * (in_h - 1)
        
        self.map1 = xs.reshape([out_h, out_w]).astype(np.float32)
        self.map2 = ys.reshape([out_h, out_w]).astype(np.float32)

    def correct_image(self, image):
        out_image = cv2.remap(image, self.map1, self.map2, cv2.INTER_LINEAR, cv2.BORDER_CONSTANT, 0)

        return out_image


def calibrate_angle(video_path, out_video_path=None, out_center_direction = [0, 120, 240], out_resolution = (1280, 720)):
    raw_video_capture = cv2.VideoCapture(video_path)
    fps = int(raw_video_capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(raw_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_bar = tqdm.tqdm(total=num_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    w, h = int(raw_video_capture.get(cv2.CV_CAP_PROP_FRAME_WIDTH)), int(raw_video_capture.get(cv2.CV_CAP_PROP_FRAME_HEIGHT))
    num_output = len(out_center_direction)
    cameras = [OminiCamera((w, h)) for _ in range(num_output)]
    
    
    return 

def main_spliter(video_path, out_video_path):
    out_resolution = (1280, 720)
    out_fov = (120, -1)
    out_center_direction = [(0, 0), (120, 0), (240, 0)]
    camera_0, camera_120, camera_240 = OminiCamera((3840, 1920)), OminiCamera((3840, 1920)), OminiCamera((3840, 1920))
    camera_0.set_output_config(out_resolution, out_center_direction[0], out_fov)
    camera_120.set_output_config(out_resolution, out_center_direction[1], out_fov)
    camera_240.set_output_config(out_resolution, out_center_direction[2], out_fov)
    raw_video_capture = cv2.VideoCapture(video_path)
    fps = int(raw_video_capture.get(cv2.CAP_PROP_FPS))
    num_frames = int(raw_video_capture.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_bar = tqdm.tqdm(total=num_frames)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_name = os.path.splitext(video_path)[0]
    
    video_name = os.path.basename(video_name)
    spk = video_name.split('-')[0]
    dir_tmp_0 = spk + '/' + spk + '-' + video_name.split('-')[1] + '/' +video_name + '-0.mp4'
    dir_tmp_120 = spk + '/' + spk + '-' + video_name.split('-')[1] + '/' +video_name + '-120.mp4'
    dir_tmp_240 = spk + '/' + spk + '-' + video_name.split('-')[1] + '/' +video_name + '-240.mp4'
    out_0 = os.path.join(out_video_path, dir_tmp_0)
    out_120 = os.path.join(out_video_path, dir_tmp_120)
    out_240 = os.path.join(out_video_path, dir_tmp_240)
    os.makedirs(os.path.dirname(out_0), exist_ok=True)
    video_0_writer = cv2.VideoWriter(out_0, fourcc, fps, out_resolution)
    video_120_writer = cv2.VideoWriter(out_120, fourcc, fps, out_resolution)
    video_240_writer = cv2.VideoWriter(out_240, fourcc, fps, out_resolution)

    while raw_video_capture.isOpened():
        ret, frame = raw_video_capture.read()
        if ret:
            image_0, image_120, image_240 = camera_0.correct_image(frame), camera_120.correct_image(frame), camera_240.correct_image(frame)
            video_0_writer.write(image_0)
            video_120_writer.write(image_120)
            video_240_writer.write(image_240)
            frames_bar.update(1)
        else:
            break
    raw_video_capture.release()
    video_0_writer.release()
    video_120_writer.release()
    video_240_writer.release()
    frames_bar.close()
    
    raw_videoclip = VideoFileClip(video_path)
    return None


def find_raw_video(root_dir):
    sub_folder = []
    raw_video_list = []
    processed_video_list = []
    while True:
        test_expression = os.path.join(root_dir, *sub_folder,'*')
        if len(glob.glob(test_expression)) > 0:
            find_expression = os.path.join(root_dir, *sub_folder, '*.mp4')
            for i in glob.glob(find_expression):
                if '-0' not in i and '-120' not in i and '-240' not in i:
                    raw_video_list.append(i)
            sub_folder.append('*')
        else:
            break
    return sorted(raw_video_list)

if __name__ == '__main__':
    parser = argparse.ArgumentParser('')
    parser.add_argument('in_video_path', type=str, default='', help='original PSCx3 dir')
    parser.add_argument('out_video_path', type=str, default='', help='processed video dir')
    args = parser.parse_args()

    raw_video_list = find_raw_video(args.in_video_path)

    for raw_video in raw_video_list:
        main_spliter(raw_video, args.out_video_path)