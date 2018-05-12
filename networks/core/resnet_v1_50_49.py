import os

from config import config as FLAGS
from networks.lib.resnet_model import create_resnet50_model

_CHECKPOINT_NAME = 'self_trained/model49/model.ckpt-112612'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = []
def func(inputs, num_classes=None, is_training=False, global_pool=True,
         output_stride=None, reuse=None, scope='resnet_v1_50_49'):
    model = create_resnet50_model()
    logits, aux_logits = model.build_network(inputs, is_training, num_classes, scope=scope, data_format='NHWC',
                                             reuse=reuse)
    return logits, aux_logits
