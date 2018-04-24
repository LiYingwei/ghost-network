import time

from config import config as FLAGS
from networks import network
from utils import *


class Model:
    def __init__(self, sess):
        self.x_input = x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
        self.y_input = y_input = tf.placeholder(tf.int32, (None,))
        logits, _ = network.model(sess, x_input, FLAGS.attack_network)
        loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y_input))
        self.grad, = tf.gradients(loss, x_input)
        self.sess = sess

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        if FLAGS.pgd:
            x = x_nat + np.random.uniform(-FLAGS.max_epsilon, FLAGS.max_epsilon, x_nat.shape)
        else:
            x = np.copy(x_nat)
        lower_bound = np.clip(x_nat - FLAGS.max_epsilon, 0, 1)
        upper_bound = np.clip(x_nat + FLAGS.max_epsilon, 0, 1)

        grads = []
        for _ in range(FLAGS.num_steps):
            for i in range(FLAGS.self_ens_num):
                noise = self.sess.run(self.grad, feed_dict={self.x_input: x, self.y_input: y})
                noise = np.array(noise) / (np.mean(np.abs(noise), axis=(1, 2, 3), keepdims=True) * 100 + 1)
                grad = grads[i] if len(grads) > i else np.zeros(shape=x_nat.shape)
                noise = FLAGS.momentum * grad + noise
                if len(grads) > i:
                    grads[i] = noise
                else:
                    grads.append(noise)
            x = np.add(x, FLAGS.step_size * np.sign(np.sum(grads, axis=0)), out=x, casting='unsafe')
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
