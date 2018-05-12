import functools
import importlib

import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import optimistic_restore


def _get_model(reuse, arg_scope, func, network_name):
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            if 'resnet_v1_50' == network_name:
                return func(images, 1000, is_training=False, reuse=reuse, scope=network_name)
            else:
                return func(images, 1001, is_training=False, reuse=reuse, scope=network_name)

    return network_fn


def _preprocess(image):
    return image * 2. - 1.


_network_initialized = {}


def model(sess, image, network_name):
    # arg_scope, func, checkpoint_path
    network_core = importlib.import_module('networks.core.' + network_name)

    global _network_initialized
    if network_name not in _network_initialized:
        _network_initialized[network_name] = False
    network_fn = _get_model(reuse=_network_initialized[network_name], arg_scope=network_core.arg_scope,
                            func=network_core.func, network_name=network_name)
    preprocessed = _preprocess(image)
    logits = tf.squeeze(network_fn(preprocessed)[0])
    predictions = tf.argmax(logits, 1)

    if not _network_initialized[network_name]:
        optimistic_restore(sess, network_core.checkpoint_path)
        _network_initialized[network_name] = True

    return logits, predictions
