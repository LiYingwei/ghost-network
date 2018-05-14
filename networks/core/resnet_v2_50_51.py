import os

from config import config as FLAGS
from networks.lib.resnet_model import create_resnet50_v2_model

_CHECKPOINT_NAME = 'self_trained_resnet_v2_50/model51/model.ckpt-46010'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = []
def func(inputs, num_classes=None, is_training=False, global_pool=True,
         output_stride=None, reuse=None, scope='resnet_v2_50_51'):
    model = create_resnet50_v2_model()
    logits, aux_logits = model.build_network(inputs, is_training, num_classes, scope=scope, data_format='NHWC',
                                             reuse=reuse)
    return logits, aux_logits
