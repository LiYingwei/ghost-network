import functools
import importlib

import tensorflow as tf
import tensorflow.contrib.slim as slim


def _get_model(reuse, arg_scope, func, network_name):
    @functools.wraps(func)
    def network_fn(images):
        with slim.arg_scope(arg_scope):
            return func(images, 1001, is_training=False, reuse=reuse, scope=network_name)

    return network_fn


def _preprocess(image):
    return image * 2. - 1.


def model(image, scope_name, label=None):
    # arg_scope, func, checkpoint_path
    network_core = importlib.import_module('networks.core.' + scope_name)

    if scope_name not in _network_initialized:
        _network_initialized[scope_name] = False

    network_fn = _get_model(reuse=_network_initialized[scope_name], arg_scope=network_core.arg_scope, func=network_core.func, network_name=scope_name)
    preprocessed = _preprocess(image)
    logits, end_points = network_fn(preprocessed)
    logits = tf.reshape(logits, shape=[-1, 1001])
    predictions = tf.argmax(logits, 1)
    if label is not None:
        acc = tf.reduce_mean(tf.cast(tf.equal(predictions, label), tf.float32))
        return acc

    return logits, predictions, end_points


_network_initialized = {}


def restore(sess, scope_name):
    network_core = importlib.import_module('networks.core.' + scope_name)
    global _network_initialized

    if (scope_name not in _network_initialized) or (not _network_initialized[scope_name]):
        ckpt_path = network_core.checkpoint_path
        optimistic_restore(sess, ckpt_path)
        _network_initialized[scope_name] = True


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
            else:
                print(var_shape, saved_shapes[saved_var_name])
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)
