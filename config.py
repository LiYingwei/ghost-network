import argparse
import collections
import inspect
import os

from easydict import EasyDict as edict

import_from = inspect.getframeinfo(inspect.getouterframes(inspect.currentframe())[1][0])[0]
eval_mode = 0 if 'eval' not in import_from else 1

config = edict(d=collections.OrderedDict())
# attack related
config.attack_network = ""
config.step_size = 1.0
config.max_epsilon = 15.0
config.num_steps = 18
config.momentum = 0.0

# eval related
config.test_network = "0145678"
config.eval_clean = False
config.val = False

# misc
config.batch_size = 10
config.report_step = 100
config.overwrite = False
config.skip = False
config.img_num = -1

# data related
# config.train_list_filename = 'data/train_list2500.txt'
config.test_list_filename = 'data/list/test_list.txt'
config.val_list_filename = 'data/list/val_list50000.txt'
config.ground_truth_file = 'data/valid_gt.csv'
config.img_dir = 'data/val_data_png/'
# config.val_img_dir = '../../data/val_data/'
config.checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints')
config.exp = 'I-FGSM'

parser = argparse.ArgumentParser(description='Process some integers.')
for key, value in config.iteritems():
    if type(value) is bool:
        parser.add_argument("--" + key, action='store_' + str(not value).lower())
    else:
        parser.add_argument("--" + key, type=type(value), default=value)
args = parser.parse_args()
for key, value in args.__dict__.iteritems():
    config[key] = value

network_pool = ["inception_v3", "inception_v4", "resnet_v2_50", "resnet_v2_101", "resnet_v2_152", "inception_resnet_v2",
                #      0               1               2               3                4                     5
                "ens3_inception_v3", "ens4_inception_v3", "ens_inception_resnet_v2",
                #      6                       7                       8
                ]

config.attack_networks = [network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index in config.attack_network]
config.test_networks = [network_pool[ord(index) - ord('a') + 10] if index >= 'a' else network_pool[int(index)] for index in config.test_network]
config.result_dir = 'result/{:s}_{:s}'.format(config.exp, config.attack_network)

if eval_mode:
    if config.eval_clean:
        if config.val:
            config.test_list_filename = config.val_list_filename
        config.result_dir = config.img_dir
else:
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    else:
        assert config.overwrite or config.skip, "{:s}".format(config.result_dir)

assert config.batch_size > 1
if config.skip:
    raise NotImplementedError
print(config)
