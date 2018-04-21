import os

from config import config as FLAGS
from networks.lib import lib_inception_resnet_v2

_CHECKPOINT_NAME = 'ens_adv_inception_resnet_v2.ckpt'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = lib_inception_resnet_v2.inception_resnet_v2_arg_scope(weight_decay=0.0)
func = lib_inception_resnet_v2.inception_resnet_v2
