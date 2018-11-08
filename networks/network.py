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
            if 'resnet_v1_50' == network_name:
                return func(images, 1000, is_training=False, reuse=reuse, scope=network_name)
            else:
                return func(images, 1001, is_training=False, reuse=reuse, scope=network_name)

    return network_fn


def _preprocess(image):
    return image * 2. - 1.


def _preprocess2(image):
    _R_MEAN = 123.68
    _G_MEAN = 116.78
    _B_MEAN = 103.94
    _CHANNEL_MEANS = [_R_MEAN, _G_MEAN, _B_MEAN]
    image = image * 255.0
    image = tf.subtract(image, _CHANNEL_MEANS)
    return image


def _preprocess3(image):
    image = image[:, :, :, ::-1]
    return image


def _preprocess4(image):
    image = image * 2. - 1.
    image = tf.image.resize_images(image, [64, 64], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return image


_network_initialized = {}

def model(sess, image, scope_name):
    # arg_scope, func, checkpoint_path
    network_name = scope_name.split('/')[0]
    ckpt_id = None if len(scope_name.split('/')) == 1 else scope_name.split('/')[1]
    network_core = importlib.import_module('networks.core.' + network_name)

    global _network_initialized
    if scope_name not in _network_initialized:
        _network_initialized[scope_name] = False
    network_fn = _get_model(reuse=_network_initialized[scope_name], arg_scope=network_core.arg_scope,
                            func=network_core.func, network_name=scope_name)
    if network_name in ['resnet_v1_50', 'resnet_v1_50_official', 'resnet_v2_50_official']:
        preprocessed = _preprocess2(image)
    elif network_name in ['resnet_50_tp']:
        preprocessed = _preprocess3(image)
    elif network_name in ['resnet_v2_50_alp']:
        preprocessed = _preprocess4(image)
    else:
        preprocessed = _preprocess(image)
    logits = tf.squeeze(network_fn(preprocessed)[0])
    predictions = tf.argmax(logits, 1)

    if not _network_initialized[scope_name]:
        try:
            ckpt_path = network_core.get_checkpoint_path(network_name, ckpt_id)
        except:
            ckpt_path = network_core.checkpoint_path
        optimistic_restore(sess, ckpt_path)
        _network_initialized[scope_name] = True

    return logits, predictions
