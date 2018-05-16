import time

from config import config as FLAGS
from networks import network
from utils import *


class Model:
    def __init__(self, sess):
        self.sess = sess
        self.x_input = x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.y_input = y_input = tf.placeholder(tf.int32, (None,))

        pre_softmax = []

        for network_name in FLAGS.attack_networks:
            logits, _ = network.model(sess, Model._input_diversity(x_input), network_name)
            pre_softmax.append(logits)

        logits_mean = tf.reduce_mean(pre_softmax, axis=0)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_mean, labels=y_input)
        self.grad = tf.gradients(loss, x_input)[0]

    @staticmethod
    def _input_diversity(input_tensor):
        if not FLAGS.input_diversity:
            return input_tensor
        rnd = tf.random_uniform((), FLAGS.image_width, FLAGS.image_resize, dtype=tf.int32)
        rescaled = tf.image.resize_images(input_tensor, [rnd, rnd], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        h_rem = FLAGS.image_resize - rnd
        w_rem = FLAGS.image_resize - rnd
        pad_top = tf.random_uniform((), 0, h_rem, dtype=tf.int32)
        pad_bottom = h_rem - pad_top
        pad_left = tf.random_uniform((), 0, w_rem, dtype=tf.int32)
        pad_right = w_rem - pad_left
        padded = tf.pad(rescaled, [[0, 0], [pad_top, pad_bottom], [pad_left, pad_right], [0, 0]], constant_values=0.)
        padded.set_shape((input_tensor.shape[0], FLAGS.image_resize, FLAGS.image_resize, 3))
        return tf.cond(tf.random_uniform(shape=[1])[0] < tf.constant(FLAGS.prob), lambda: padded, lambda: input_tensor)

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if FLAGS.pgd:
            x = x_nat + np.random.uniform(-FLAGS.max_epsilon, FLAGS.max_epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)
        lower_bound = np.clip(x_nat - FLAGS.max_epsilon, 0, 1)
        upper_bound = np.clip(x_nat + FLAGS.max_epsilon, 0, 1)

        grad = None
        for _ in range(FLAGS.num_steps):
            noise = self.sess.run(self.grad, feed_dict={self.x_input: x, self.y_input: y})
            noise = np.array(noise) / np.maximum(1e-12, np.mean(np.abs(noise), axis=(1, 2, 3), keepdims=True))
            grad = 0 if grad is None else grad
            grad = FLAGS.momentum * grad + noise

            x = np.add(x, FLAGS.step_size * np.sign(grad), out=x, casting='unsafe')
            x = np.clip(x, lower_bound, upper_bound)

        return x


if __name__ == '__main__':
    xs = load_data(FLAGS.test_list_filename)
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
            time_predict = time_used / (batch_index + 1) * (5000 / FLAGS.batch_size - batch_index - 1)
            print('{} images have been processed, {:.2}h used, {:.2}h need'.
                  format(image_index, time_used / 3600, time_predict / 3600))
