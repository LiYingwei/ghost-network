import time

from config import config as FLAGS
from networks import network
from utils import *

import numpy as np
import scipy.stats as st
from tensorpack.models import (Conv2D, MaxPooling, AvgPooling)
from tensorpack.tfutils import argscope


def gkern(kernlen=21, nsig=3):
    """Returns a 2D Gaussian kernel array."""

    interval = (2 * nsig + 1.) / (kernlen)
    x = np.linspace(-nsig - interval / 2., nsig + interval / 2., kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    # kernel_raw[:] = 1
    kernel = kernel_raw / kernel_raw.sum()
    return kernel


def shift_stack(theta, feature_size=3):
    l_shape = theta.get_shape().as_list()
    padding_size = (feature_size - 1) // 2
    theta = tf.pad(theta, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
    theta_stack = []
    for h in range(feature_size):
        for w in range(feature_size):
            theta_stack.append(theta[:, h:h + l_shape[1], w:w + l_shape[2], :])
    theta = tf.concat(theta_stack, axis=3)
    return theta


def local_non_local_op(gradient, feature, embed, softmax, kernel_size=5):
    feature = tf.image.resize_images(feature, [299, 299])
    with argscope([Conv2D, MaxPooling, AvgPooling], data_format='channels_first'):
        gradient = tf.transpose(gradient, [0, 3, 1, 2])
        feature = tf.transpose(feature, [0, 3, 1, 2])
        l_shape = gradient.get_shape().as_list()
        if embed:
            theta = Conv2D('embedding_theta', feature, l_shape[1]/2, 1, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            phi = Conv2D('embedding_phi', feature, l_shape[1]/2, 1, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        else:
            theta, phi = feature, feature
        g_orig = g = gradient

        padding_size = (kernel_size - 1) // 2
        phi = tf.pad(phi, [[0, 0], [0, 0], [padding_size, padding_size], [padding_size, padding_size]])
        g = tf.pad(g, [[0, 0], [0, 0], [padding_size, padding_size], [padding_size, padding_size]])
        out = tf.zeros_like(gradient)

        if softmax:
            f = []
            for h in range(kernel_size):
                for w in range(kernel_size):
                    f.append(tf.reduce_sum(tf.multiply(theta, phi[:,:,h:h+l_shape[2],w:w+l_shape[3]]), axis=1, keepdims=True))
            f = tf.concat(f, axis=1)
            f = f / tf.sqrt(tf.cast(l_shape[1]/2, tf.float32))
            f = tf.nn.softmax(f, axis=1)
            for h in range(kernel_size):
                for w in range(kernel_size):
                    out += tf.multiply(f[:,w+h*kernel_size:w+h*kernel_size+1,:,:], g[:,:,h:h+l_shape[2],w:w+l_shape[3]])
        else:
            for h in range(kernel_size):
                for w in range(kernel_size):
                    f = tf.reduce_sum(tf.multiply(theta, phi[:,:,h:h+l_shape[2],w:w+l_shape[3]]), axis=1, keepdims=True)
                    out += tf.multiply(f, g[:,:,h:h+l_shape[2],w:w+l_shape[3]])
            out = out / (kernel_size**2)
        out = tf.transpose(out, [0, 2, 3, 1])
        return out

def non_local_op(gradient, feature, embed, softmax, maxpool, avgpool):
    # feature = tf.image.resize_images(feature, [299, 299])
    with argscope([Conv2D, MaxPooling, AvgPooling], data_format='channels_first'):
        # change to channels first
        gradient = tf.transpose(gradient, [0, 3, 1, 2])
        feature = tf.transpose(feature, [0, 3, 1, 2])
        if embed:
            n_in = feature.get_shape().as_list()[1]
            theta = Conv2D('embedding_theta', feature, n_in / 2, 1, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
            phi = Conv2D('embedding_phi', feature, n_in / 2, 1, strides=1, kernel_initializer=tf.random_normal_initializer(stddev=0.01))
        else:
            theta, phi = feature, feature
        g_orig = g = gradient
        # whether apply pooling function
        assert (avgpool == 1 or maxpool == 1)
        if maxpool > 1:
            phi = MaxPooling('pool_phi', phi, pool_size=maxpool, stride=maxpool)
            g = MaxPooling('pool_g', g, pool_size=maxpool, stride=maxpool)
        if avgpool > 1:
            phi = AvgPooling('pool_phi', phi, pool_size=avgpool, stride=avgpool)
            g = AvgPooling('pool_g', g, pool_size=avgpool, stride=avgpool)
        # flatten tensors
        theta_flat = tf.reshape(theta, [tf.shape(theta)[0], tf.shape(theta)[1], -1])
        phi_flat = tf.reshape(phi, [tf.shape(phi)[0], tf.shape(phi)[1], -1])
        g_flat = tf.reshape(g, [tf.shape(g)[0], tf.shape(g)[1], -1])
        theta_flat.set_shape([theta.shape[0], theta.shape[1], theta.shape[2] * theta.shape[3] if None not in theta.shape[2:] else None])
        phi_flat.set_shape([phi.shape[0], phi.shape[1], phi.shape[2] * phi.shape[3] if None not in phi.shape[2:] else None])
        g_flat.set_shape([g.shape[0], g.shape[1], g.shape[2] * g.shape[3] if None not in g.shape[2:] else None])
        # Compute production
        f = tf.matmul(theta_flat, phi_flat, transpose_a=True)

        # import pdb; pdb.set_trace()

        if softmax:
            f = f / tf.sqrt(float(n_in / 2))
            f = tf.nn.softmax(f)
        else:
            f = f / tf.cast(tf.shape(f)[-1], f.dtype)

        out = tf.matmul(g_flat, f, transpose_b=True)
        ret = tf.reshape(out, tf.shape(g_orig))
        ret = tf.transpose(ret, [0, 2, 3, 1])
        return ret


class Model:
    def __init__(self, sess):
        self.sess = sess
        self.x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.y_input = tf.placeholder(tf.int32, (None,))
        self.x_inputs = []
        self.grads = []

        assert len(FLAGS.attack_networks) == 1
        network_name = FLAGS.attack_networks[0]
        logits, _, endpoints = network.model(sess, self.x_input, network_name)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y_input)
        self.grad = tf.gradients(loss, self.x_input)[0]

        # if FLAGS.local_non_local:
        #     self.grad = Model.local_non_local(self.grad, FLAGS.kernel_size, endpoints['Conv2d_1a_3x3'])
        if FLAGS.gaussian:
            self.grad = Model.gaussian(self.grad, FLAGS.kernel_size, self.x_input)
        if FLAGS.non_local:
            with tf.variable_scope('non_local'):
                self.grad = local_non_local_op(self.grad, endpoints['Conv2d_2b_3x3'], embed=True, softmax=True)

        self.non_local_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='non_local')
        sess.run(tf.variables_initializer(self.non_local_variables))

        # TODO: def
        G = tf.get_default_graph()
        with G.gradient_override_map({"Sign": "Identity"}):
            x_adv = self.x_input + FLAGS.step_size * tf.sign(self.grad)
        x_adv = tf.clip_by_value(x_adv, 0, 1)
        logits, _, _ = network.model(sess, x_adv, 'resnet_v2_50')
        self.loss_to_optimize = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self.y_input)

        with tf.variable_scope('train_op'):
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(-self.loss_to_optimize, var_list=self.non_local_variables)
            self.train_op_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='train_op')
        sess.run(tf.variables_initializer(self.train_op_variables))


    # @staticmethod
    # def local_non_local(l, kernel_size, feature):
    #     feature = tf.image.resize_images(feature, [299, 299])
    #     l_shape = l.get_shape().as_list()
    #     theta, phi = feature, feature
    #     # theta, phi = shift_stack(theta, FLAGS.feature_size), shift_stack(phi, FLAGS.feature_size)
    #     # phi = phi[:, :, :, ::-1]
    #     # import pdb; pdb.set_trace()
    #     g_orig = g = l
    #
    #     padding_size = (kernel_size - 1) // 2
    #     phi = tf.pad(phi, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
    #     g = tf.pad(g, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
    #     out = tf.zeros_like(l)
    #
    #     for h in range(kernel_size):
    #         for w in range(kernel_size):
    #             f = tf.reduce_sum(tf.multiply(theta, phi[:, h:h + l_shape[1], w:w + l_shape[2], :]), axis=3, keepdims=True)
    #             out += tf.multiply(f, g[:, h:h + l_shape[1], w:w + l_shape[2], :])
    #             # import pdb; pdb.set_trace()
    #     out = out / (kernel_size ** 2)
    #     return out

    @staticmethod
    def gaussian(l, kernel_size, image):
        f = gkern(kernel_size)

        l_shape = l.get_shape().as_list()
        g_orig = g = l
        padding_size = (kernel_size - 1) // 2
        g = tf.pad(g, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]])
        out = tf.zeros_like(l)

        # import pdb; pdb.set_trace()

        for h in range(kernel_size):
            for w in range(kernel_size):
                out += f[h, w] * g[:, h:h + l_shape[1], w:w + l_shape[2], :]
                # import pdb; pdb.set_trace()
        out = out / (kernel_size ** 2)
        return out

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        x = np.copy(x_nat)
        lower_bound = np.clip(x_nat - FLAGS.max_epsilon, 0, 1)
        upper_bound = np.clip(x_nat + FLAGS.max_epsilon, 0, 1)

        grad = self.sess.run(self.grad, feed_dict={self.x_input: x, self.y_input: y})
        x = np.add(x, FLAGS.step_size * np.sign(grad), out=x, casting='unsafe')
        x = np.clip(x, lower_bound, upper_bound)

        return x

    def step(self, x, y):
        loss, _ = self.sess.run([self.loss_to_optimize, self.train_op], feed_dict={self.x_input: x, self.y_input: y})
        return loss


if __name__ == '__main__':
    xs = load_data_with_checking(FLAGS.test_list_filename, FLAGS.result_dir) if FLAGS.skip else load_data(FLAGS.test_list_filename)
    ys = get_label(xs, FLAGS.ground_truth_file)
    x_batches = split_to_batches(xs, FLAGS.batch_size)
    y_batches = split_to_batches(ys, FLAGS.batch_size)
    sess = tf.Session()

    model = Model(sess)
    start = time.time()
    for epoch in range(50):
        loss_avg = 0.

        x_batches = split_to_batches(xs, FLAGS.batch_size)
        y_batches = split_to_batches(ys, FLAGS.batch_size)
        # import pdb;pdb.set_trace()
        for batch_index, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
            # import pdb; pdb.set_trace()
            images = load_images(x_batch, FLAGS.test_img_dir)
            labels = y_batch
            loss = np.sum(model.step(images, labels))
            loss_avg += loss
            # print(np.sum(loss))
            if epoch % 5 == 0:
                advs = model.perturb(images, labels)
                save_images(advs, x_batch, FLAGS.result_dir)

            # image_index = batch_index * FLAGS.batch_size
            # if image_index % FLAGS.report_step == 0:
            #     time_used = time.time() - start
            #     time_predict = time_used / (batch_index + 1) * (len(xs) / FLAGS.batch_size - batch_index - 1)
            #     print('{} images have been processed, {:.2}h used, {:.2}h need'.
            #           format(image_index, time_used / 3600, time_predict / 3600))
        print("epoch {:d}: loss={:f}".format(epoch, loss_avg / 500))
        # print(sess.run(model.non_local_variables[0])[0][0][:10])
