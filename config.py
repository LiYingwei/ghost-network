import argparse
import collections
from inspect import currentframe
import os

from easydict import EasyDict as edict

frame = currentframe().f_back
while frame.f_code.co_filename.startswith('<frozen'):
    frame = frame.f_back
import_from = frame.f_code.co_filename
eval_mode = 0 if 'eval' not in import_from else 1

config = edict(d=collections.OrderedDict())
# attack related
config.attack_network = ""
config.step_size = 1.0
config.max_epsilon = 8.0
config.num_steps = 10
config.momentum = 0.0

# ghost network related
config.optimal = False
config.random_range = 0.0
config.keep_prob = 1.0

# eval related
config.test_network = "234501687"
config.eval_clean = False
config.val = False
config.GPU_ID = '0'

# misc
config.batch_size = 10
config.report_step = 100
config.overwrite = False
config.skip = False
config.img_num = -1

# data related
config.test_list_filename = 'data/list/test_list5000.txt'
config.val_list_filename = 'data/list/val_list50000.txt'
config.ground_truth_file = 'data/valid_gt.csv'
config.img_dir = 'data/val_data/'
config.checkpoint_path = os.path.join('data', 'checkpoints')
config.exp = 'I-FGSM'

parser = argparse.ArgumentParser(description='Process some integers.')
for key, value in config.items():
    if type(value) is bool:
        parser.add_argument("--" + key, action='store_' + str(not value).lower())
    else:
        parser.add_argument("--" + key, type=type(value), default=value)
args = parser.parse_args()
for key, value in args.__dict__.items():
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
        config.random_range = 0.0
        config.keep_prob = 1.0
        config.optimal = False
else:
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    else:
        assert config.overwrite or config.skip, "{:s}".format(config.result_dir)

os.environ['CUDA_VISIBLE_DEVICES'] = config.GPU_ID

if config.skip:
    raise NotImplementedError
print(config)
