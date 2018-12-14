import functools
import importlib

import tensorflow as tf
import tensorflow.contrib.slim as slim
from config import config as FLAGS

from utils import optimistic_restore


def _get_model(reuse, arg_scope, func, network_name):
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse, scope=network_name)

    return network_fn


def _preprocess(image):
    return image * 2. - 1.

_network_initialized = {}

def model(sess, image, scope_name):
    # arg_scope, func, checkpoint_path
    network_name = scope_name.split('/')[0]
    ckpt_id = None if len(scope_name.split('/')) == 1 else scope_name.split('/')[1]
    network_core = importlib.import_module('networks.core.' + network_name)

    global _network_initialized
    if scope_name not in _network_initialized:
        _network_initialized[scope_name] = False
    network_fn = _get_model(reuse=_network_initialized[scope_name], arg_scope=network_core.arg_scope, func=network_core.func, network_name=scope_name)
    preprocessed = _preprocess(image)
    logits, end_points = network_fn(preprocessed)
    logits = tf.squeeze(logits)
    predictions = tf.argmax(logits, 1)

    if not _network_initialized[scope_name]:
        try:
            ckpt_path = network_core.get_checkpoint_path(network_name, ckpt_id)
        except:
            ckpt_path = network_core.checkpoint_path
        optimistic_restore(sess, ckpt_path)
        _network_initialized[scope_name] = True

    return logits, predictions, end_points
