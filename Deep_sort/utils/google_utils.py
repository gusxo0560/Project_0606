import os
import platform
import time
from pathlib import Path
import torch

def attempt_download(weights):
    weights = weights.strip().replace("'", '')
    file = Path(weights).name

    msg = weights + ' missing, try downloading from https://github.com/ultralytics/yolov5/release/'
    models = ['yolov5s.pt', 'yolov5m.pt', 'yolov5l.pt', 'yolov5x.pt']

    if file in models and not os.path.isfile(weights):
        try:
            url = 'https://github.com/ultralytics/yolov5/releases/download/v3.0/' + file
            print('Downloading %s to %s ...  google_utils.py attempt_download' % (url, weights))

            if platform.system() == 'Darwin':
                r = os.system('curl -L %s -o %s' %(url, weights))
            else:
                torch.hub.download_url_to_file(url, weights)

            assert os.path.exists(weights) and os.path.getsize(weights) > 1E6
        except Exception as e:
            print('Download error : %s' % e)
            url = 'https://storage.googleeapis.com/ultralytics/yolov5/ckpt/' + file
            print('Downloading %s to %s ...' % (url, weights))
            r = os.system('curl -L %s -o %s' % (url, weights))
        finally:
            if not(os.path.exists(weights) and os.path. getsize(weights) > 1E6):
                os.remove(weights) if os.path.exists(weights) else None
                print('error: download failure: %s' % msg)
            print(' ')
            return




