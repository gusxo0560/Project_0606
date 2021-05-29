import cv2
import torch
import time
import torchvision
import numpy as np

def xywh2xyxy(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2

    return y

def xyxy2xywh(x):
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = (x[:, 0] + x[:, 2]) / 2
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2
    y[:, 2] = x[:, 2] + x[:, 0]
    y[:, 3] = x[:, 3] + x[:, 1]

    return y

def box_iou(box1, box2):
    def box_area(box):
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.t())
    area2 = box_area(box2.t())

    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)

def clip_coords(boxes, img_shape):
    boxes[:, 0].clamp_(0, img_shape[1])
    boxes[:, 1].clamp_(0, img_shape[0])
    boxes[:, 2].clamp_(0, img_shape[1])
    boxes[:, 3].clamp_(0, img_shape[0])

def scale_coords(img1_shape,coords, img0_shape, ratio_pad=None):
    if ratio_pad is None:
        gain = max(img1_shape) / max(img0_shape)
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]
    coords[:, [1, 3]] -= pad[1]
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)

    return coords


def non_max_suppression(prediction, conf_thres=0.1, iou_thres=0.6, fast=False,  classes=None, agnostic=False):
    if prediction.dtype is torch.float16:
        prediction = prediction.float()

    nc = prediction[0].shape[1] - 5
    xc = prediction[..., 4] > conf_thres

    min_wh, max_wh = 2, 4096
    max_det = 300
    time_limit = 10.0
    redundant = True
    fast |= conf_thres > 0.001
    multi_label = nc > 1

    if fast:
        merge = False
    else:
        merge = True

    t = time.time()
    output = [None] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue
        x[:, 5:] *= x[:, 4:5]

        box = xywh2xyxy(x[:, :4])

        if multi_label:
            i, j = (x[:, 5:] >conf_thres).nonzero().t()
            x = torch.cat((box[i], x[i, j + 5, None], j[:, None].float()), 1)

        else:
            conf, j = x[:, 5:].max(1, keepdim=True)
            x = torch.cat((box, conf, j.float()), 1[conf.view(-1) > conf_thres])

        n = x.shape[0]
        if not n:
            continue

        c = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + c, x[:, 4]
        i = torchvision.ops.boxes.nms(boxes, scores, iou_thres)

        if i .shape[0] > max_det:
            i = i[:max_det]

        if merge and (1 < n < 3E3):
            try:
                iou = box_iou(boxes[i], boxes) > iou_thres
                weights = iou * scores[None]
                x[i, :4] = torch.mm(weights, x[:, :4]).float() / weights.sum(1, keepdim=True)
                if redundant:
                    i = i[iou.sum(1) > 1]

            except:
                print('utils.py non_max_suppression except: ',x, i, x.shape, i.shape)
                pass

        output[xi] = x[i]
        if (time.time() - t) >  time_limit:
            break

    return output

def apply_classifier(x, model, img, im0):
    im0 = [im0] if isinstance(im0, np.ndarray) else im0
    for i, d in enumerate(x):
        if d is not None and len(d):
           d = d.clone()

           b = xyxy2xywh(d[:, :4])
           b[:, 2:] = b[:, 2:].max(1)[0].unsqueeze(1)
           b[:, 2:] = b[:, 2:] * 1.3 + 30
           d[:, :4] = xywh2xyxy(b).long()

           scale_coords(img.shape[2:], d[:, :4], im0[i].shape)

           pred_cls1 = d[:, 5].long()
           ims = []
           for j, a in enumerate(d):
               cutout = im0[i][int(a[1]):int(a[3]), int(a[0]):int(a[2])]
               im = cv2.resize(cutout, (224, 224))

               im = im[:, :, ::-1].transpose(2, 0, 1)
               im = np.ascontiguousarray(im, dtype=np.float32)
               im /= 255.0
               ims.append(im)
           pred_cls2 = model(torch.Tensor(ims).to(d.device)).argmax(1)
           x[i] = x[i][pred_cls1 == pred_cls2]

    return x

def plot_dots_on_people(x, img):
    thickness = -1
    color = [0, 255, 0]
    center = ((int(x[2]) + int(x[0])) // 2 , (int(x[3]) + int(x[1])) // 2)
    radius = 6
    cv2.circle(img, center, radius, color, thickness)

    return img



