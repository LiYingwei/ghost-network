import os

from config import config as FLAGS
from nets import inception_resnet_v2 as nets

_CHECKPOINT_NAME = 'inception_resnet_v2_2016_08_30.ckpt'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = nets.inception_resnet_v2_arg_scope(weight_decay=0.0)
func = nets.inception_resnet_v2
