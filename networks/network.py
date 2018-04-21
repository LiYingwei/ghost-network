import functools
import importlib

import tensorflow as tf
import tensorflow.contrib.slim as slim

from utils import optimistic_restore


def _get_model(reuse, arg_scope, func):
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse)

    return network_fn


def _preprocess(image):
    return image * 2. - 1.


_network_initialized = False


def model(sess, image, network_name):
    # arg_scope, func, checkpoint_path
    network_core = importlib.import_module('networks.core.' + network_name)

    global _network_initialized
    network_fn = _get_model(reuse=_network_initialized, arg_scope=network_core.arg_scope, func=network_core.func)
    preprocessed = _preprocess(image)
    logits = tf.reshape(network_fn(preprocessed)[0], (-1, 1001))
    predictions = tf.argmax(logits, 1)

    if not _network_initialized:
        optimistic_restore(sess, network_core.checkpoint_path)
        _network_initialized = True

    return logits, predictions
