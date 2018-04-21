import os

import tensorflow.contrib.slim.nets as nets

from config import config as FLAGS

_CHECKPOINT_NAME = 'inception_v3.ckpt'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)
arg_scope = nets.inception.inception_v3_arg_scope(weight_decay=0.0)
func = nets.inception.inception_v3
