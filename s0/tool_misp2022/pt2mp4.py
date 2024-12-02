import torch
import cv2
import numpy as np

data = torch.load('/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt-minggao/examples/misp2022/s0/data_v/training_video_segment/pt/S218_A225_S218235236237238_F8N_Near_218_007688-007868.pt')

frames = data.numpy()

frame_rate = 25
height, width = frames.shape[1:3]

fourcc=cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter('/train33/sppro/permanent/hangchen2/pandora/egs/misp2024-mmmt-minggao/examples/misp2022/s0/data_v/dev_video_segment/output_video.mp4', fourcc, frame_rate, (width, height))

for frame in frames:
    frame = np.uint8(frame)
    if frame.shape[-1] == 3:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    video_writer.write(frame)

video_writer.release()