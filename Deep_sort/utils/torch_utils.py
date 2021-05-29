import math
import os
import time
import logging

from copy import deepcopy

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

logger = logging.getLogger(__name__)
#
# def init_seeds(seed = 0):
#     torch.manual_seed(seed)
#
#     if seed == 0:
#         cudnn.deterministic = True
#         cudnn.benchmark = False
#     else:
#         cudnn.deterministic = False
#         cudnn.benchmark = True
#
#
#
# def is_parallel(model):
#     return type(model) in (nn.parallel.DataParallel, nn.parallel.DistributedDataParallel)
#

def select_device(device='', batch_size=None):
    cpu_request = device.lower() == 'cpu'
    if device and not cpu_request:
        os.environ['CUDA_VISIBLE_DEVICES'] = device

        assert torch.cuda.is_available(), 'torch_utils.py select_device /// CUDA unavailable, invalid device %s requested ' % device

    cuda = False if cpu_request else torch.cuda.is_available()
    if cuda:
        c = 1024 ** 2
        ng = torch.cuda.device_count()
        if ng > 1 and batch_size:
            assert batch_size % ng == 0, 'torch_utils.py select_device///batch-size %g not multiple of GPU count %g' % (batch_size, ng)

        x = [torch.cuda.get_device_properties(i) for i in range(ng)]
        s = 'Using CUDA '

        for i in range(0, ng):
            if i == 1:
                s = ' ' * len(s)
            logger.info('%sdevice%g _CudaDeviceProperties(anme="%s", total_memory=%dMB)' % (s, i, x[i].name, x[i].total_memory / c))

    else:
        logger.info('Using CPU')

    logger.info('')

    return torch.device('cuda:0' if cuda else 'cpu')


def load_classifier(name='resnet1010', n=2):
    model = models.__dict__[name](pretrained=True)

    input_sie = [3, 224, 224]
    input_space = 'RGB'
    input_range = [0, 1]
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    for x in[input_sie, input_space, input_range, mean, std]:
        print('torch_utils.py load_classifier : ', x + '=', eval(x))

    filters = model.fc.weight.shape[1]
    model.fc.bias = nn.Parameter(torch.zeros(n), requires_grad=True)
    model.fc.weight = nn.Parameter(torch.zeros(n, filters), requires_grad=True)
    model.fc.out_features = n

    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    return time.time()

def fuse_conv_and_bn(conv, bn):
    with torch.no_grad():
        fusedconv = nn.Conv2d(conv.in_channels,
                              conv.out_channels,
                              kernel_size=conv.kernel_size,
                              stride=conv.stride,
                              padding=conv.padding,
                              bias=True).to(conv.weight.device)

        w_conv = conv.weight.clone().view(conv.out_channels, -1)
        w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
        fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.size()))

        b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
        b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
        fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

        return fusedconv

def model_info(model, verbose=False):
    n_p = sum(x.numel() for x in model.parameters())
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)
    if verbose:
        print('%s %40s %9s %12s %20s %10s %10s' %('layer', 'name', 'gradient', 'parameters', 'shape', 'mu','sigma'))
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            print('%5g %40s %9s %12g %20s %10.3g %10.3g' %(i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std()))
    try:
        from thop import profile
        flops = profile(deepcopy(model), inputs=(torch.zeros(1, 3, 64, 64), ), verbose=False)[0] / 1E9 * 2
        fs = ', %.1f GELOPS' % (flops * 100)
    except:
        fs = ''

    logger.info('Model Summary: %g layers, %g parameters, %g gradients%s' % (len(list(model.parameters())), n_p, n_g, fs))

def scale_img(img,ratio=1.0, same_shape=False):
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)
        if not same_shape:
            gs = 32
            h, w = [math.ceil(x * ratio /gs) * gs for x in (h, w)]

        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)

def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momenum = 0.03
        elif t in [nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True