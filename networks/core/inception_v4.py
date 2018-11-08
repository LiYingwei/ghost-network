import os

from config import config as FLAGS
import tensorflow.contrib.slim.nets as nets

_CHECKPOINT_NAME = 'inception_v4.ckpt'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = nets.inception.inception_v4_arg_scope(weight_decay=0.0)
func = nets.inception.inception_v4
