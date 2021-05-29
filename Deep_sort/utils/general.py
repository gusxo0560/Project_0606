import glob
import math
import os
import random
import shutil
import subprocess
import time
import logging
from contextlib import contextmanager
from copy import copy
from pathlib import Path
import platform

import cv2

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import yaml
from scipy.cluster.vq import kmeans
from scipy.signal import butter, filtfilt
from tqdm import tqdm


torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})
matplotlib.rc('font', **{'size': 11})

cv2.setNumThreads(0)

def check_anchor_order(m):
    a = m.anchor_grid.prod(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da.sign() != ds.sign():
        print('utils.general.py check_anchor_order : Reversing anchor order')
        m.anchors[:] = m.anchors.filp(0)
        m.anchor_grid[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)

def make_divisible(x, divisor):
    return math.ceil(x / divisor) * divisor

def set_logging(rank=-1):
    logging.basicConfig(format='%(message)s', level=logging.INFO if rank in [-1, 0] else logging.WARN)

def check_file(file):
    if os.path.isfile(file) or file == '':
        return file
    else:
        files = glob.glob('./**/' + file, recursive=True)
        assert len(file), 'utils.general.py check_file : File Not Found: %s' % file
        return files[0]