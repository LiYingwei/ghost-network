import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops, tensor_shape, tensor_util
from tensorflow.python.eager import context
from tensorflow.python.ops import math_ops, nn_ops, variable_scope, array_ops, random_ops
import numbers


def _get_noise_shape(x, noise_shape):
    # If noise_shape is none return immediately.
    if noise_shape is None:
        return array_ops.shape(x)

    try:
        # Best effort to figure out the intended shape.
        # If not possible, let the op to handle it.
        # In eager mode exception will show up.
        noise_shape_ = tensor_shape.as_shape(noise_shape)
    except (TypeError, ValueError):
        return noise_shape

    if x.shape.dims is not None and len(x.shape.dims) == len(noise_shape_.dims):
        new_dims = []
        for i, dim in enumerate(x.shape.dims):
            if noise_shape_.dims[i].value is None and dim.value is not None:
                new_dims.append(dim.value)
            else:
                new_dims.append(noise_shape_.dims[i].value)
        return tensor_shape.TensorShape(new_dims)

    return noise_shape


def dropout_fix(x, keep_prob, noise_shape=None, seed=None, name=None):  # pylint: disable=invalid-name
    """Computes dropout.

    With probability `keep_prob`, outputs the input element scaled up by
    `1 / keep_prob`, otherwise outputs `0`.  The scaling is so that the expected
    sum is unchanged.

    By default, each element is kept or dropped independently.  If `noise_shape`
    is specified, it must be
    [broadcastable](http://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)
    to the shape of `x`, and only dimensions with `noise_shape[i] == shape(x)[i]`
    will make independent decisions.  For example, if `shape(x) = [k, l, m, n]`
    and `noise_shape = [k, 1, 1, n]`, each batch and channel component will be
    kept independently and each row and column will be kept or not kept together.

    Args:
      x: A floating point tensor.
      keep_prob: A scalar `Tensor` with the same type as x. The probability
        that each element is kept.
      noise_shape: A 1-D `Tensor` of type `int32`, representing the
        shape for randomly generated keep/drop flags.
      seed: A Python integer. Used to create random seeds. See
        @{tf.set_random_seed}
        for behavior.
      name: A name for this operation (optional).

    Returns:
      A Tensor of the same shape of `x`.

    Raises:
      ValueError: If `keep_prob` is not in `(0, 1]` or if `x` is not a floating
        point tensor.
    """
    with ops.name_scope(name, "dropout", [x]) as name:
        x = ops.convert_to_tensor(x, name="x")
        if not x.dtype.is_floating:
            raise ValueError("x has to be a floating point tensor since it's going to"
                             " be scaled. Got a %s tensor instead." % x.dtype)
        if isinstance(keep_prob, numbers.Real) and not 0 < keep_prob <= 1:
            raise ValueError("keep_prob must be a scalar tensor or a float in the "
                             "range (0, 1], got %g" % keep_prob)

        # Early return if nothing needs to be dropped.
        if isinstance(keep_prob, float) and keep_prob == 1:
            return x
        if context.executing_eagerly():
            if isinstance(keep_prob, ops.EagerTensor):
                if keep_prob.numpy() == 1:
                    return x
        else:
            keep_prob = ops.convert_to_tensor(
                keep_prob, dtype=x.dtype, name="keep_prob")
            keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())

            # Do nothing if we know keep_prob == 1
            if tensor_util.constant_value(keep_prob) == 1:
                return x

        noise_shape = _get_noise_shape(x, noise_shape)

        # uniform [keep_prob, 1.0 + keep_prob)
        random_tensor = keep_prob

        with tf.variable_scope("fix_dropout", reuse=tf.AUTO_REUSE):
            np_random = np.random.uniform(0, 1, x.shape[1:])
            random_result = tf.get_variable('random_{:d}'.format(np.random.randint(0, 2147483647)), shape=np_random.shape, initializer=tf.constant_initializer(np_random))
            # random_result = random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)

        random_tensor += random_result
        # 0. if [keep_prob, 1.0) and 1. if [1.0, 1.0 + keep_prob)
        binary_tensor = math_ops.floor(random_tensor)
        ret = math_ops.div(x, keep_prob) * binary_tensor
        if not context.executing_eagerly():
            ret.set_shape(x.get_shape())
        return ret
