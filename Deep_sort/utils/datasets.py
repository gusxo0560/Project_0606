import cv2
from threading import Thread
import numpy as np
import os
import time
import module
class LoadStreams:

    def __init__(self, sources='streams.txt', img_size=640):
        self.video = module.CV_FILES
        self.mode = 'images'
        self.img_size = img_size

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        print('datasets.py LoadStreams -sources : ', self.sources)
        for i, s in enumerate(sources):
            print('detection.py LoadStreams - for : %g/%g: %s... ' % (i + 1, n, s), end='')
            cap = cv2.VideoCapture(self.video)
            assert cap.isOpened(), 'datasets.py LoadStreams -cv2.VideoCapture error : Faild to open %s' % s

            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS) % 100
            _, self.imgs[i] = cap.read()
            thread = Thread(target=self.update, args=([i, cap]), daemon=True)
            print('datasets.py LoadStreams : success(%g * %g at %.2f FPS).' % (w, h, fps))
            thread.start()
        print('')

        s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)
        self.rect = np.unique(s, axis=0).shape[0] == 1

        if not self.rect:
            print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        n = 0
        while cap.isOpened():
            n += 1
            cap.grab()
            if n == 4:
                _, self.imgs[index] = cap.retrieve()
                n = 0
            time.sleep(0.04) #늦추는곳 default=0.04
    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1
        img0 = self.imgs.copy()
        if cv2.waitKey(1) == ord('q'):
            cv2.destroyAllWindows()
            raise StopIteration

        img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]
        img = np.stack(img, 0)
        img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)

        return self.sources, img, img0, None

    def __len__(self):
        return 0


def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):

    shape = img.shape[:2]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0 / 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value = color)

    return img, ratio, (dw, dh)

