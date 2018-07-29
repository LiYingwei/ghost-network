import os

from networks.lib import lib_inception_v3

from config import config as FLAGS

_CHECKPOINT_NAME = 'ens3_adv_inception_v3.ckpt'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = lib_inception_v3.inception_v3_arg_scope(weight_decay=0.0)
func = lib_inception_v3.inception_v3
