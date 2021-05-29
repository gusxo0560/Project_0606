import cv2
import numpy as np
import os

class Detection(object):

    def __init__(self, tlwh, confidence, feature, clses):
        self.tlwh = np.asarray(tlwh, dtype=np.float)
        self.confidence = float(confidence)
        self.feature = np.asarray(feature, dtype=np.float32)
        self.clses = int(clses)

    def to_tlbr(self):
        ret = self.tlwh.copy()
        ret[2:] += ret[:2]

        return ret

    def to_xyah(self):
        ret = self.tlwh.copy()
        ret[:2] += ret[2:] / 2
        ret[2] /= ret[3]

        return ret

