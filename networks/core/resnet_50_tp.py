from tensorpack.tfutils.tower import TowerContext

from config import config as FLAGS
from networks.lib.lib_resnet_50_tp import *
import tensorflow as tf

# _CHECKPOINT_NAME = 'ImageNet-ResNet50.ckpt'
_CHECKPOINT_NAME = 'tp/model-14820'
checkpoint_path = os.path.join(
    FLAGS.checkpoint_path,
    _CHECKPOINT_NAME
)

arg_scope = []


def func(inputs, num_classes=1001, is_training=False, global_pool=True, output_stride=None, reuse=None, scope='tp1'):
    with TowerContext("", is_training=False):
        # with tf.variable_scope('tp1'):
        model = Model(
            depth=50,
            mode='preact'
        )
        # necessary preprocess
        # inputs = tf.transpose(inputs, [0, 3, 1, 2])
        output = model.get_logits(inputs)
        return output, None
