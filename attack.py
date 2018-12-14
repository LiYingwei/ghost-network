import time

from config import config as FLAGS
from networks import network
from utils import *

import numpy as np
import scipy.stats as st


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


class Model:
    def __init__(self, sess):
        self.sess = sess
        self.x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.y_input = tf.placeholder(tf.int32, (None,))
        self.x_inputs = []
        self.grads = []

        pre_softmax = []
        # assert len(FLAGS.attack_networks) == 1
        for network_name in FLAGS.attack_networks:
            x_input = tf.identity(self.x_input)
            self.x_inputs.append(x_input)
            logits, _, endpoints = network.model(sess, x_input, network_name)
            pre_softmax.append(logits)

        logits_mean = tf.reduce_mean(pre_softmax, axis=0)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_mean, labels=self.y_input)
        self.grad = tf.gradients(loss, self.x_input)[0]

        # if FLAGS.local_non_local:
        #     self.grad = Model.local_non_local(self.grad, FLAGS.kernel_size, endpoints['Conv2d_1a_3x3'])
        if FLAGS.gaussian:
            self.grad = Model.gaussian(self.grad, FLAGS.kernel_size, self.x_input)

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

        grad = None
        for _ in range(FLAGS.num_steps):
            noise = self.sess.run(self.grad, feed_dict={self.x_input: x, self.y_input: y})
            noise = np.array(noise) / np.maximum(1e-12, np.mean(np.abs(noise), axis=(1, 2, 3), keepdims=True))
            grad = noise if grad is None else FLAGS.momentum * grad + noise

            # grads = self.sess.run(self.grads, feed_dict={self.x_input: x, self.y_input: y})
            # grad = np.zeros(grads[0].shape)
            # for g in grads:
            #     grad += g / np.linalg.norm(g)

            x = np.add(x, FLAGS.step_size * np.sign(grad), out=x, casting='unsafe')
            x = np.clip(x, lower_bound, upper_bound)

        return x


if __name__ == '__main__':
    xs = load_data_with_checking(FLAGS.test_list_filename, FLAGS.result_dir) if FLAGS.skip else load_data(FLAGS.test_list_filename)
    ys = get_label(xs, FLAGS.ground_truth_file)
    x_batches = split_to_batches(xs, FLAGS.batch_size)
    y_batches = split_to_batches(ys, FLAGS.batch_size)
    sess = tf.Session()

    model = Model(sess)
    start = time.time()
    for batch_index, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
        images = load_images(x_batch, FLAGS.test_img_dir)
        labels = y_batch
        advs = model.perturb(images, labels)
        save_images(advs, x_batch, FLAGS.result_dir)

        image_index = batch_index * FLAGS.batch_size
        if image_index % FLAGS.report_step == 0:
            time_used = time.time() - start
            time_predict = time_used / (batch_index + 1) * (len(xs) / FLAGS.batch_size - batch_index - 1)
            print('{} images have been processed, {:.2}h used, {:.2}h need'.
                  format(image_index, time_used / 3600, time_predict / 3600))
