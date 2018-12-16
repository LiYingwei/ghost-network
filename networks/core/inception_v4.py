import os

from config import config as FLAGS
from nets import inception_v4 as nets

_CHECKPOINT_NAME = 'inception_v4.ckpt'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = nets.inception_v4_arg_scope(weight_decay=0.0)
func = nets.inception_v4
