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
config.random_range = 0.0
config.keep_prob = 1.0
config.attack_network = "resnet_v2_152"
config.pgd = False
config.cw = False
config.FGSM = False
config.restart = False
config.cont = False
config.self_ens_num = 1
config.momentum = 0.0
config.input_diversity = False
config.eval_clean = False
config.val = False
config.optimal = False
config.optimal_dropout = False
config.test_network_id = "873201465"
config.dropout_fix = False
config.fix = False
config.max_epsilon = 15.0
config.eval_dir = None # don't change this line

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument("--random_range", type=float, default=config.random_range)
parser.add_argument("--keep_prob", type=float, default=config.keep_prob)
parser.add_argument("--attack_network", type=str, default=config.attack_network)
parser.add_argument("--test_network_id", type=str, default=config.test_network_id)
parser.add_argument("--pgd", action='store_true')
parser.add_argument("--cw", action='store_true')
parser.add_argument("--FGSM", action='store_true')
parser.add_argument("--restart", action='store_true')
parser.add_argument("--cont", action='store_true')
parser.add_argument("--input_diversity", action='store_true')
parser.add_argument("--self_ens_num", type=int, default=config.self_ens_num)
parser.add_argument("--momentum", type=float, default=config.momentum)
parser.add_argument("--dropout_fix", action='store_true')
parser.add_argument("--fix", action='store_true')
parser.add_argument("--max_epsilon", type=float, default=config.max_epsilon)
parser.add_argument("--eval_dir", type=str)

parser.add_argument("--eval_clean", action='store_true')
parser.add_argument("--val", action='store_true')
parser.add_argument("--optimal", action='store_true')
parser.add_argument("--optimal_dropout", action='store_true')

args = parser.parse_args()
for key, value in args.__dict__.iteritems():
    config[key] = value

config.max_epsilon /= 255.0
config.step_size = 1.0 / 255 if not config.FGSM else config.max_epsilon
config.num_steps = int(min(config.max_epsilon * 255 + 4, 1.25 * config.max_epsilon * 255)) if not config.FGSM else 1
config.report_step = 100

attack_networks_pool = ["inception_v3", "inception_v4", "inception_resnet_v2",  # 0-2
                        "resnet_v2_152", "ens3_inception_v3", "ens4_inception_v3",  # 3-5
                        "ens_inception_resnet_v2", "resnet_v2_101", "resnet_v2_50",  # 6-8
                        "resnet_v2_50_official", "resnet_v2_50_38", "resnet_v2_50_49",  # 9-b
                        "resnet_v2_50_51", "resnet_v2_50_138", "resnet_v2_50_205", "resnet_v2_50_fix",  # c-f
                        "resnet_v2_101_fix", "resnet_v2_152_fix", "inception_resnet_v2_fix",  # g-i
                        "resnet_v2_50_dropout", "resnet_v2_50_scale_res", "adv_inception_v3", # j,k,l
                        "resnet_50_tp", "ens_inception_resnet_v2_fix", "resnet_v2_50_alp"]  # m, n, o

if 'ensemble' in import_from or config.eval_clean:
    config.attack_networks = []
    for index in config.attack_network:
        if index >= 'a':
            i = ord(index) - ord('a') + 10
        else:
            i = int(index)
        config.attack_networks.append(attack_networks_pool[i])

test_network = ["inception_v3", "inception_v4", "inception_resnet_v2",  # 0-2
                "resnet_v2_152", "ens3_inception_v3", "ens4_inception_v3",  # 3-5
                "ens_inception_resnet_v2", "resnet_v2_101", "resnet_v2_50",  # 6-8
                "resnet_v2_50_official", "resnet_v2_50_38", "resnet_v2_50_49",  # 9-b
                "resnet_v2_50_51", "resnet_v2_50_138", "resnet_v2_50_205", "resnet_50_tp"]  # c,d,e,f

# test_network["87320146"] = ["resnet_v2_50", "resnet_v2_101", "resnet_v2_152",
#                             "inception_resnet_v2", "inception_v3", "inception_v4",
#                             "ens3_inception_v3", "ens_inception_resnet_v2"]

config.test_network = []
for index in config.test_network_id:
    if index >= 'a':
        i = ord(index) - ord('a') + 10
    else:
        i = int(index)
    config.test_network.append(test_network[i])

config.test_list_filename = 'data/test_list5000.txt'  # if not config.cw else 'data/test_list1000.txt'
config.val_list_filename = 'data/val_list50000.txt'
config.ground_truth_file = 'data/valid_gt.csv'
config.test_img_dir = 'data/test_data/'
config.val_img_dir = '../../data/val_data/'
config.checkpoint_path = os.path.join(os.path.dirname(__file__), 'checkpoints')

# DI-FGSM param
config.image_width = 299
config.image_resize = 330
config.prob = 0.5

assert not config.pgd
random_range_str = "{:.3f}".format(config.random_range) if not config.optimal else "optimal"
if config.pgd:
    config.result_dir = 'PGD_{:s}_{:s}'.format(config.attack_network, random_range_str)
elif config.FGSM:
    config.result_dir = 'FGSM_{:s}_{:s}'.format(config.attack_network, random_range_str)
elif config.cw:
    config.result_dir = 'CW_{:s}_{:s}'.format(config.attack_network, random_range_str)
else:
    config.result_dir = 'I-FGSM_{:s}_{:s}'.format(config.attack_network, random_range_str)

if config.dropout_fix:
    config.result_dir += "_dropout_fix{:.3f}".format(config.keep_prob)

if config.fix:
    config.result_dir += "_fix{:.3f}".format(config.keep_prob)

if config.self_ens_num > 1:
    config.result_dir += "_slfens{:d}".format(config.self_ens_num)
    print("deprecated")
    assert 0

if config.keep_prob < 1.0:
    config.result_dir += "_keep_prob{:.3f}".format(config.keep_prob)

if config.momentum > 0.0:
    config.result_dir += "_momentum{:.2f}".format(config.momentum)

if config.input_diversity:
    config.result_dir += "_D"

config.base_dir = "result"
config.base_dir2 = "archived"
config.result_dir = os.path.join(config.base_dir, config.result_dir)
config.target_dir = os.path.join(config.base_dir2, config.result_dir)

if eval_mode == 1:
    if config.eval_clean:
        if config.val:
            config.test_img_dir = config.val_img_dir
            config.test_list_filename = config.val_list_filename
        config.result_dir = config.test_img_dir
        config.test_network = config.attack_networks
    else:
        config.random_range = 0.0
        config.keep_prob = 1.0
        config.optimal = False
        config.optimal_dropout = False
    config.batch_size = 32
else:
    config.batch_size = 25
    if not os.path.exists(config.result_dir):
        os.makedirs(config.result_dir)
    else:
        assert config.restart or config.cont, "{:s}".format(config.result_dir)

assert config.batch_size > 1

print(config)
