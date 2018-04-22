import inspect

import_from = inspect.getframeinfo(inspect.getouterframes(inspect.currentframe())[1][0])[0]
attack_mode = 0
eval_mode = 0
if 'label_smoothing' in import_from:
    attack_mode = 1
if 'eval' in import_from:
    eval_mode = 1

import argparse
import os

from easydict import EasyDict as edict

config = edict()
config.random_range = 0.1  # will be change to 0.0 if mode is eval
config.attack_network = "resnet_v2_152"
config.ensemble_num = 30
config.smoothing_factor = 0.9  # will be change to 1.0 if mode is eval

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--random_range", type=float, default=config.random_range)
parser.add_argument("--attack_network", type=str, default=config.attack_network)
parser.add_argument("--ensemble_num", type=int, default=config.ensemble_num)
parser.add_argument("--smoothing_factor", type=float, default=config.smoothing_factor)

args = parser.parse_args()
for key, value in args.__dict__.iteritems():
    config[key] = value

config.max_epsilon = 15.0 / 255
config.step_size = 1.0 / 255
config.num_steps = int(min(config.max_epsilon * 255 + 4, 1.25 * config.max_epsilon * 255))
config.report_step = 100

config.test_network = ["inception_v3", "inception_v4", "inception_resnet_v2", "resnet_v2_152", "ens3_inception_v3",
                       "ens4_inception_v3", "ens_inception_resnet_v2", "resnet_v2_101", "resnet_v2_50"]
# config.test_network = ["ens_inception_resnet_v2", "inception_resnet_v2", "resnet_v2_152", "resnet_v2_101",
#                        "resnet_v2_50"]

config.test_list_filename = '../data/FlorianProject/test_list_full.txt'
# config.test_list_filename = '../data/FlorianProject/test_list.txt'
config.ground_truth_file = '../data/FlorianProject/valid_gt.csv'
config.test_img_dir = '../data/FlorianProject/test_data/'
config.checkpoint_path = os.path.join(os.path.dirname(__file__), 'data')

if attack_mode == 0:
    config.result_dir = 'result/random_{:s}_{:.3f}_{:d}'. \
        format(config.attack_network, config.random_range, config.ensemble_num)
elif attack_mode == 1:
    config.result_dir = 'result/smoothing_{:s}_{:.3f}_{:d}_{:.2f}'. \
        format(config.attack_network, config.random_range, config.ensemble_num, config.smoothing_factor)
else:
    assert 0

if eval_mode == 1:
    config.random_range = 0.0
    config.smoothing_factor = 1.0

if not os.path.exists(config.result_dir):
    os.makedirs(config.result_dir)

print(config)
