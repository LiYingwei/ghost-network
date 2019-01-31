import os

from nets import resnet_v2_50 as nets

from config import config as FLAGS

_CHECKPOINT_NAME = 'resnet_v2_50.ckpt'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = nets.resnet_arg_scope(weight_decay=0.0)
func = nets.resnet_v2_50
