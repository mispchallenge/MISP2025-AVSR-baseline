import cv2
import numpy as np
import time
import onnxruntime
import argparse
import os
from yolox_process import byte_preprocess, byte_postprocess
from yolov7_process import yolov7_preprocess, yolov7_postprocess

from byte_tracker.byte_tracker import BYTETracker
from rmdeep_tracker.deep_sort import DeepSort
from Track_torch_develop.KCF.kcftracker import KCFTracker
from Track_torch_develop.ocsort_tracker.ocsort import OCSort

def detect_inference(ori_img, onnx_session, detect_model, input_size = [608, 1088], track_target='perosn'):
    if 'yolox' in detect_model:    
        img, ratio = byte_preprocess(ori_img, input_size)
        ort_inputs = {onnx_session.get_inputs()[0].name: img[None, :, :, :]}
        output = onnx_session.run(None, ort_inputs)
        dets = byte_postprocess(output[0], ratio, input_size)
        return dets    # [[x1, y1, x2, y2, score]...]    -> array
    else:
        img, shapes = yolov7_preprocess(ori_img)
        ort_inputs = {onnx_session.get_inputs()[0].name: img.cpu().numpy()}
        output = onnx_session.run(None, ort_inputs)
        det_body, det_head, det_face = yolov7_postprocess(output[0], shapes)
        if track_target == 'person':
            return det_body # [[x1, y1, x2, y2, score], ...]   -> list
        elif track_target == 'face':
            return det_face # [[x1, y1, x2, y2, score], ...]   -> list
        else:
            return det_head # [[x1, y1, x2, y2, score], ...]   -> list

def get_image_list(path):
    files = []
    with open(path, 'r') as f1:
        for file_name in f1.readlines():
            if file_name != None:
                files.append(file_name.strip("\n"))
    return files

def get_video_list(path):
    files = []
    with open(path, 'r') as f1:
        for file_name in f1.readlines():
            if file_name != None:
                files.append(file_name.strip("\n"))
    return files

def createFile(filePath):
    if not os.path.exists(filePath):
        try:
            os.mkdir(filePath)
        except Exception as e:
            os.makedirs(filePath)

def get_color(idx):
    idx = idx * 3
    color = ((37 * idx) % 255, (17 * idx) % 255, (29 * idx) % 255)
    return color

def draw_boxes(results, output_path):
    createFile(output_path)
    for dets in results:
        det = dets.strip(';').split(';')
        img = cv2.imread(det[1])
        store_name = output_path + '/' + os.path.basename(det[1])
        if len(det) > 2:
            for item in det[2:]:
                temp = item.split('\t')
                id = temp[0]
                x = int(temp[1])
                y = int(temp[2])
                w = int(temp[3])
                h = int(temp[4])
                color = get_color(int(id))
                cv2.rectangle(img,(x, y, w, h), color, 3)
                cv2.putText(img, id, (x, y-10), cv2.FONT_HERSHEY_PLAIN, 2, (0, 0, 255), 2)
        cv2.imwrite(store_name, img)

def main(args):
    if args.input_path.endswith('txt'):
        image_list = get_image_list(args.input_path)
    else:
        image_list = get_video_list(args.input_path)  # video input，待实现

    if args.detect_model == 'yolox_tiny':     # 默认yolox
        onnx_detect_model = "/work2/cv1/smgong/track/Track_torch_develop/onnx_model/bytetrack_yolox_tiny.onnx"
    elif args.detect_model == 'yolox_x':     # 需要改下 输入(input_size)
        onnx_detect_model = "/work2/cv1/smgong/track/Track_torch_develop/onnx_model/bytetrack_yolox_x.onnx"
    else:
        onnx_detect_model = '/work2/cv1/smgong/track/Track_torch_develop/onnx_model/yolov7-tiny_dynamic.onnx'

    # 使用 ONNX Runtime 运行模型
    onnx_detect_session = onnxruntime.InferenceSession(onnx_detect_model)

    if args.track_mode == 'deepsort':
        if args.fast_reid == 'fast_reid':
            onnx_reid_model = '/work2/cv1/smgong/track/Track_torch_develop/onnx_model/fastreid.onnx'
        elif args.fast_reid == 'csk':
            onnx_reid_model = '/work2/cv1/smgong/track/Track_torch_develop/onnx_model/head_shoulder.onnx'
        elif args.fast_reid == 'align':
            onnx_reid_model = '/work2/cv1/smgong/track/Track_torch_develop/onnx_model/face_align_dynamic.onnx'
        else:
            onnx_reid_model = '/work2/cv1/smgong/track/Track_torch_develop/onnx_model/face_recog.onnx'
        onnx_reid_session = onnxruntime.InferenceSession(onnx_reid_model)
        use_Tracker = DeepSort()      # 先用默认配置，后续可以将几个参数弄成配置项
    elif args.track_mode == 'bytetrack':
        use_Tracker = BYTETracker(buffer_size = 50)   # 只开放了这一个参数
    elif args.track_mode == 'ocsort':
        use_Tracker = OCSort(use_byte=True)        # 先用默认配置，后续可以将几个参数弄成配置项
    else:
        use_Tracker = KCFTracker()    # kcf
    

    # import pdb;pdb.set_trace()
    record_detect_time = []
    record_track_time = []
    results = []
    for frame_id, image in enumerate(image_list, 0):
        save_cont = '1;' + image + ';'
        img = cv2.imread(image)
        time1 = time.time()
        dets = detect_inference(img, onnx_detect_session, args.detect_model, args.input_size, args.track_target)
        time2 = time.time()
        record_detect_time.append(time2 - time1)
        time3 = time.time()
        img_height, img_width = img.shape[:2]
        if dets is not None:
            if args.track_mode == 'bytetrack':
                online_targets = use_Tracker.update(dets, [img_height, img_width], args.input_size)    # xyxy
                for t in online_targets:
                    tlwh = t.tlwh
                    tid = t.track_id
                    if tlwh[2] * tlwh[3] > 10:
                        save_cont += str(tid) + '\t' + str(max(0, int(tlwh[0]))) + '\t' + str(max(0, int(tlwh[1]))) + '\t' + str(int(tlwh[2])) + '\t' + str(int(tlwh[3])) + ';'
            elif args.track_mode == 'ocsort':
                online_targets = use_Tracker.update(dets, [img_height, img_width], args.input_size)    # xyxy
                for t in online_targets:
                    tlwh = [t[0], t[1], t[2] - t[0], t[3] - t[1]]
                    tid = int(t[4])
                    if tlwh[2] * tlwh[3] > 10:
                        save_cont += str(tid) + '\t' + str(max(0, int(tlwh[0]))) + '\t' + str(max(0, int(tlwh[1]))) + '\t' + str(int(tlwh[2])) + '\t' + str(int(tlwh[3])) + ';'
            elif args.track_mode == 'deepsort':
                if dets.shape[0] == 0:
                    xyxy = dets
                    confidences = dets
                else:                        
                    xyxy = dets[:, 0:4]
                    confidences = dets[:, 4]
                online_targets = use_Tracker.update(xyxy, confidences, img, onnx_reid_session, args.fast_reid)    # xyxy
                for t in online_targets:
                    save_cont += str(t[-1]) + '\t' + str(max(0, int(t[0]))) + '\t' + str(max(0, int(t[1]))) + '\t' + str(int(t[2] - t[0])) + '\t' + str(int(t[3] - t[1])) + ';'
            else:   
                if dets.shape[0] == 0:
                    xywh = dets
                else:
                    xyxy = dets[:, 0:4]
                    xywh = xyxy.copy()
                    xywh[:, 2] = xyxy[:,2] - xyxy[:,0]
                    xywh[:, 3] = xyxy[:,3] - xyxy[:,1]
                is_detect = False
                if frame_id % 15 ==0:
                    is_detect = True
                online_targets = use_Tracker.update_and_kf_newer(xywh, img, is_detect)    # xywh
                # online_targets = use_Tracker.update(xywh, img, frame_id)    # xywh
                for t in online_targets:
                    save_cont += str(t[0]) + '\t' + str(max(0, int(t[1]))) + '\t' + str(max(0, int(t[2]))) + '\t' + str(int(t[3])) + '\t' + str(int(t[4])) + ';'
        time4 = time.time()
        record_track_time.append(time4 - time3)

        results.append(save_cont)

        print(frame_id, ' / ', len(image_list))
        
    with open(args.save_path, 'w+') as ff:
        for i in range(len(results)):
            ff.write(results[i].strip(';') + '\n')

    print('detect time: ', sum(record_detect_time) / len(record_detect_time) )
    print('track time: ', sum(record_track_time) / len(record_track_time) )

    if args.visual:
        draw_boxes(results, args.output_path)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="/train13/cv1/smgong/data/FaceDetect/try/track_test_data/track_far_3.txt")
    parser.add_argument("--detect_model", type=str, default="yolox_tiny")
    parser.add_argument("--input_size", nargs='+', type=int, default=[608, 1088])
    parser.add_argument("--track_mode", type=str, default="bytetrack")
    parser.add_argument("--track_target", type=str, default="person")
    parser.add_argument("--fast_reid", type=str, default='fast_reid', help='person track use fast_reid, face track use csk or face_recog')
    parser.add_argument("--visual", action="store_true")
    parser.add_argument("--output_path", type=str, default="/work2/cv1/smgong/track/Track_torch_develop/visual", help='folder for storing visual result')
    parser.add_argument("--save_path", type=str, default="/work2/cv1/smgong/track/Track_torch_develop/test_result.txt")
    parser.add_argument("--cpu", dest="use_cuda", action="store_false", default=True)
    return parser.parse_args()


if __name__ == '__main__':
    print('face/head_track only support yolov7, person_track support both!!!')
    args = parse_args()
    main(args)

    print('all is over!!!')
