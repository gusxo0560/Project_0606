import os
import cv2
import random
import shutil
import numpy as np
# *.py Files
from scipy.spatial import distance

import module
from models.experimental import attempt_load
from deep_sort_pytorch.utils.parser import get_config
from deep_sort_pytorch.deep_sort import DeepSort
from base_camera import BaseCamera

from utils import torch_utils

from utils.datasets import LoadStreams
from utils.utils import *
from utils.torch_utils import load_classifier, select_device
from log import Logger

import module

class Camera(BaseCamera):
    video_source = module.CV_FILES
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

    def __init__(self):
        if os.environ.get('OPENCV_CAMERA_SOURCE////openCV_.py __init__'):
            print('success')
            Camera.set_video_source(int(os.environ['OPENCV_CAMERA_SOURCE']))
        super(Camera, self).__init__()

        print("don't success")

    @staticmethod
    def set_video_source(source):
        Camera.video_source = source

    @staticmethod
    def bbox_rel(*xyxy):
        bbox_left = min([xyxy[0].item(), xyxy[2].item()])
        bbox_top = min([xyxy[1].item(), xyxy[3].item()])
        bbox_w = abs(xyxy[0].item() - xyxy[2].item())
        bbox_h = abs(xyxy[1].item() - xyxy[3].item())

        x_c = (bbox_left + bbox_w / 2)
        y_c = (bbox_top + bbox_h / 2)

        w = bbox_w
        h = bbox_h

        return x_c, y_c, w, h

    @staticmethod
    def compute_color_for_labels(label):
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in Camera.palette]
        return tuple(color)

    @staticmethod
    def draw_boxes(img, bbox, cls_names, scores, identities=None, offset=(0,0)):
        for i, box in enumerate(bbox):
            x1, y1, x2, y2 = [int(i) for i in box]
            x1 += offset[0]
            x2 += offset[0]
            y1 += offset[1]
            y2 += offset[1]

            id = int(identities[i]) if identities is not None else 0
            color = Camera.compute_color_for_labels(id)
            label = '%d %s %d ' % (id, cls_names[i], scores[i])
            label += '%'
            print('{0}출력중--- openCV_.py  draw_boxes ======================'.format(id))

            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 1)
            cv2.rectangle(img, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), color, -1)
            cv2.putText(img, label, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)
        #
        img = Camera.draw_lines(img, bbox)

        return img

    @staticmethod
    def draw_lines(img, bbox):
        # 검색
        result1 = []
        for i, box in enumerate(bbox):
            rr = [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
            for i_, box_ in enumerate(bbox[i + 1:]):
                rr2 = [(box_[0] + box_[2]) / 2, (box_[1] + box_[3]) / 2]
                if Camera.Checking(rr, rr2):
                    result1.append([rr, rr2])

        result1 = np.array(result1, dtype=np.int)
        if len(result1) != 0:
            for i in range(len(result1)):
                r1, r2 = result1[i][0], result1[i][1]
                img = cv2.line(img, (r1[0], r1[1]), (r2[0], r2[1]), (0, 0, 255), 2, cv2.LINE_AA)

        return img

    @staticmethod
    def Checking(a, b):
        dist = distance.euclidean(a, b)
        calibration = (a[1] + b[1]) / 2
        if 0 < dist < 0.25 * calibration:
            return True
        else:
            return False

    @staticmethod
    def frames():
        logger = Logger()
        print('camera initialize.... - openCV_.py frames')
        camera = cv2.VideoCapture(Camera.video_source)
        if not camera.isOpened():
            raise RuntimeError('Could not start camera//// openCv_.py frames')

        out, weights, imgsz = '.inference/output', module.weights_File, 640
        source = '0'

        cfg = get_config()
        cfg.merge_from_file('./deep_sort_pytorch/configs/deep_sort.yaml')
        # deepsort = DeepSort(cfg.DEEPSORT.REID_CKPT,
        #         max_dist = cfg.DEEPSORT.MAX_DIST, min_confidence= cfg.DEEPSORT.MIN_CONFIDENCE,
        #         nms_max_overlap=cfg.DEEPSORT.NMS_MAX_OVERLAP, max_iou_distacne=cfg.DEEPSORT.MAX_IOU_DISTANCE,
        #         max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT, nn_budget=cfg.DEEPSORT.NN_BUDGET, use_cuda=True)
        #
        deepsort = DeepSort("deep_sort_pytorch/deep_sort/deep/checkpoint/ckpt.t7", 0.2, 0.3, 0.5, 0.7, 70, 3, 100)
        device = torch_utils.select_device()

        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        model = attempt_load(weights, map_location=device)
        model.to(device).eval()

        '''//////////////////////////////////////////'''
        classify = False
        if classify:
            modelc = load_classifier(name='resnet101', n = 2)
            modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=device)['model'])
            modelc.to(device).eval()
        '''//////////////////////////////////////////'''

        half = False and device.type != 'cpu'

        if half:
            model.half()

        dataset = LoadStreams(source, img_size=imgsz)
        names = model.names if hasattr(model, 'names') else model.modules.names

        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        t0 = time.time()
        img = torch.zeros((1, 3, imgsz, imgsz), device=device)
        _ = model(img.half() if half else img) if device.type != 'cpu' else None
        for frame_idx, (path, img, im0s, vid_cap) in enumerate(dataset):
            img = torch.from_numpy(img).to(device)
            img = img.half() if half else img.float()
            img /= 255.0
            if img.ndimension() == 3:
                img = img.unsqueeze(0)

            t1 = torch_utils.time_synchronized()
            pred = model(img, augment=False)[0]

            pred = non_max_suppression(pred, 0.4, 0.5, fast=True, classes=None, agnostic=False)

            t2 = torch_utils.time_synchronized()

            if classify:
                pred = apply_classifier(pred, modelc, img, im0s)

            for i, det in enumerate(pred):
                p, s, im0 = path[i], '%g: ' % i, im0s[i].copy()
                s += '%gx%g ' % img.shape[2:]
                if det is not None and len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                    for c in det[:, -1].detach().unique():
                        n = (det[:, -1] == c).sum()
                        s +='%g %s, ' % (n, names[int(c)])

                    bbox_xywh =[]
                    confs = []
                    clses = []
                    i  = 0
                    for *xyxy, conf, cls in reversed(det):
                        label = '%s %.2f' % (names[int(cls)], conf)
                        print('openCV_.py frames : labels --> {0}'.format(label))
                        if label is not None:
                            if (label.split())[0] == 'person':
                                i+=1
                                x_c, y_c, bbox_w, bbox_h = Camera.bbox_rel(*xyxy)
                                obj = [x_c, y_c, bbox_w, bbox_h]
                                print('label ', i, ' : ', obj)
                                bbox_xywh.append(obj)
                                confs.append([conf.item()])
                                clses.append([cls.item()])
                                #$logger.info('openCV_.py frames- : This process Person --> {0}'.format(os.getpid))
                                plot_dots_on_people(xyxy, im0)

                try:
                    xywhs = torch.Tensor(bbox_xywh)
                    confss = torch.Tensor(confs)
                    clses = torch.Tensor(clses)
                    outputs = deepsort.update(xywhs, confss, clses, im0)

                    if len(outputs) > 0:
                        bbox_tlwh = []
                        bbox_xyxy = outputs[:, :4]
                        print('bbox_xyxy : ', bbox_xyxy)
                        identities = outputs[:, 4]
                        clses = outputs[:, 5]
                        scores = outputs[:, 6]
                        stays = outputs[:, 7]
                        Camera.draw_boxes(im0, bbox_xyxy, [names[i] for i in clses], scores, identities)


                    print('%s Done. (%.3fs)' % (s, t2 - t1))

                except :
                    print(' 아무도 없어요 ')

            cv2.imshow('img', im0)







