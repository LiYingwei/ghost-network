import inspect

import_from = inspect.getframeinfo(inspect.getouterframes(inspect.currentframe())[1][0])[0]
attack_mode = 0
eval_mode = 0

if 'eval' in import_from:
    eval_mode = 1

import argparse
import os

from easydict import EasyDict as edict

config = edict()
config.random_range = 0.1  # will be change to 0.0 if mode is eval
config.attack_network = "resnet_v2_152"
config.pgd = False
config.FGSM = False
config.restart = False
config.self_ens_num = 1
config.momentum = 0.0
config.input_diversity = False

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--random_range", type=float, default=config.random_range)
parser.add_argument("--attack_network", type=str, default=config.attack_network)
parser.add_argument("--pgd", action='store_true')
parser.add_argument("--FGSM", action='store_true')
parser.add_argument("--restart", action='store_true')
parser.add_argument("--input_diversity", action='store_true')
parser.add_argument("--self_ens_num", type=int, default=config.self_ens_num)
parser.add_argument("--momentum", type=float, default=config.momentum)

parser.add_argument("--eval_clean", action='store_true')

args = parser.parse_args()
for key, value in args.__dict__.iteritems():
    config[key] = value

config.max_epsilon = 15.0 / 255
config.step_size = 1.0 / 255 if not config.FGSM else config.max_epsilon
config.num_steps = int(min(config.max_epsilon * 255 + 4, 1.25 * config.max_epsilon * 255)) if not config.FGSM else 1
config.report_step = 100

attack_networks_pool = ["inception_v3", "inception_v4", "inception_resnet_v2", "resnet_v2_152", "ens3_inception_v3",
                        "ens4_inception_v3", "ens_inception_resnet_v2", "resnet_v2_101", "resnet_v2_50"]

if 'ensemble' in import_from:
    config.attack_networks = []
    for index in config.attack_network:
        i = int(index)
        config.attack_networks.append(attack_networks_pool[i])

config.test_network = ["inception_v3", "inception_v4", "inception_resnet_v2", "resnet_v2_152", "ens3_inception_v3",
                       "ens4_inception_v3", "ens_inception_resnet_v2", "resnet_v2_101", "resnet_v2_50"]
config.test_list_filename = 'data/test_list5000.txt'
config.ground_truth_file = 'data/valid_gt.csv'
config.test_img_dir = 'data/test_data/'
config.checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints')

# DI-FGSM param
config.image_width = 299
config.image_resize = 330
config.prob = 0.5

if config.pgd:
    config.result_dir = 'PGD_{:s}_{:.3f}'.format(config.attack_network, config.random_range)
elif config.FGSM:
    config.result_dir = 'FGSM_{:s}_{:.3f}'.format(config.attack_network, config.random_range)
else:
    config.result_dir = 'I-FGSM_{:s}_{:.3f}'.format(config.attack_network, config.random_range)

if config.self_ens_num > 1:
    config.result_dir += "_slfens{:d}".format(config.self_ens_num)

if config.momentum > 0.0:
    config.result_dir += "_momentum{:.2f}".format(config.momentum)

if config.input_diversity:
    config.result_dir += "_D"

config.base_dir = "result"
config.base_dir2 = "archived"
config.result_dir = os.path.join(config.base_dir, config.result_dir)
config.target_dir = os.path.join(config.base_dir2, config.result_dir)

if eval_mode == 1:
    config.random_range = 0.0
    config.batch_size = 128
    if args.eval_clean:
        config.result_dir = config.test_img_dir
else:
    config.batch_size = 16
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    else:
        assert config.restart, "{:s}".format(config.result_dir)

print(config)
