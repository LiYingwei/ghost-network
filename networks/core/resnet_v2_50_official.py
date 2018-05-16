import os

import tensorflow as tf

from config import config as FLAGS
from networks.lib import imagenet_main

_CHECKPOINT_NAME = 'self_trained_resnet_v2_50/official_model/model.ckpt-250200'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = []


def func(inputs, num_classes=1001, is_training=False, global_pool=True, output_stride=None, reuse=None, scope=''):
    model = imagenet_main.ImagenetModel(
        resnet_size=50,
        data_format='channels_first',
        resnet_version=2,
        dtype=tf.float32
    )
    # inputs = tf.transpose(inputs, [0, 3, 1, 2])
    output = model(inputs, training=is_training)
    return output, None
