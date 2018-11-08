import time

from config import config as FLAGS
from networks import network
from utils import *


class Model:
    def __init__(self, sess):
        self.sess = sess
        self.x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.y_input = tf.placeholder(tf.int32, (None,))
        self.x_inputs = []
        self.grads = []

        pre_softmax = []
        for network_name in FLAGS.attack_networks:
            x_input = tf.identity(self.x_input)
            self.x_inputs.append(x_input)
            logits, _ = network.model(sess, x_input, network_name)
            pre_softmax.append(logits)

        logits_mean = tf.reduce_mean(pre_softmax, axis=0)
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_mean, labels=self.y_input)
        self.grad = tf.gradients(loss, self.x_input)[0]

        for nid, network_name in enumerate(FLAGS.attack_networks):
            self.grads.append(tf.gradients(loss, self.x_inputs[nid])[0])

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        x = np.copy(x_nat)
        lower_bound = np.clip(x_nat - FLAGS.max_epsilon, 0, 1)
        upper_bound = np.clip(x_nat + FLAGS.max_epsilon, 0, 1)

        # grad = None
        for _ in range(FLAGS.num_steps):
            grad = self.sess.run(self.grad, feed_dict={self.x_input: x, self.y_input: y})

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
