import os
import yaml
from easydict import EasyDict as edict

class YamlParser(edict):
    def __init__(self, cfg_dict=None, config_file=None):
        if cfg_dict is None:
            cfg_dict = {}

        if config_file is not None:
            assert(os.path.isfile(config_file))
            with open(config_file, 'r') as fo:
                cfg_dict.update(yaml.load(fo.read()))

        super(YamlParser, self).__init__(cfg_dict)

    def merge_from_file(self, config_file):
        with open(config_file, 'r') as fo:
            print('parser.py merge_from_file ///file names : {0}'.format(config_file))
            self.update(yaml.load(fo.read()))

    def merge_from_dict(self, config_dict):
        self.update(config_dict)

def get_config(config_file=None):
    return YamlParser(config_file=config_file)


if __name__ == '__main__':
    cfg = YamlParser(config_file='../configs/yolov5l.yaml')
    cfg.merge_from_file('../configs/deep_sort.yaml')

    import ipdb

    ipdb.set_trace()
