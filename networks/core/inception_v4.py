import os

from config import config as FLAGS
from networks.lib import lib_inception_v4

_CHECKPOINT_NAME = 'inception_v4.ckpt'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = lib_inception_v4.inception_v4_arg_scope(weight_decay=0.0)
func = lib_inception_v4.inception_v4
