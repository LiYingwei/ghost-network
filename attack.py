import tensorflow as tf
from tensorpack import (BatchData)
from tqdm import tqdm

from config import config as FLAGS
from data import PNGDataFlow, save_images
from networks import network


class Attacker:
    def __init__(self, sess):
        self.sess = sess
        self.step_size = FLAGS.step_size / 255.0
        self.max_epsilon = FLAGS.max_epsilon / 255.0
        # Prepare graph
        batch_shape = [FLAGS.batch_size, 299, 299, 3]
        self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_max = tf.clip_by_value(self.x_input + self.max_epsilon, 0., 1.0)
        x_min = tf.clip_by_value(self.x_input - self.max_epsilon, 0., 1.0)

        # y = tf.constant(np.zeros([FLAGS.batch_size]), tf.int64)
        self.y_input = tf.placeholder(tf.int64, shape=batch_shape[0])
        i = tf.constant(0)
        grad = tf.zeros(shape=batch_shape)
        self.x_adv, _, _, _, _, _ = tf.while_loop(self.stop, self.graph, [self.x_input, self.y_input, i, x_max, x_min, grad])
        self.restore()

    def graph(self, x, y, i, x_max, x_min, grad):
        logits_list = []
        for network_name in FLAGS.attack_networks:
            logits, _, endpoints = network.model(x, network_name)
            logits_list.append(logits)

        logits_mean = tf.reduce_mean(logits_list, axis=0)
        # pred = tf.argmax(logits_mean, axis=1)
        # first_round = tf.cast(tf.equal(i, 0), tf.int64)
        # y = first_round * pred + (1 - first_round) * y

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_mean, labels=y)
        noise = tf.gradients(loss, x)[0]
        noise = noise / (tf.reduce_mean(tf.abs(noise), [1, 2, 3], keepdims=True) + 1e-12)
        noise = FLAGS.momentum * grad + noise
        x = x + self.step_size * tf.sign(noise)
        x = tf.clip_by_value(x, x_min, x_max)
        i = tf.add(i, 1)
        return x, y, i, x_max, x_min, noise

    @staticmethod
    def stop(x, y, i, x_max, x_min, grad):
        return tf.less(i, FLAGS.num_steps)

    def perturb(self, images, labels):
        adv_images = sess.run(self.x_adv, feed_dict={self.x_input: images, self.y_input: labels})
        return adv_images

    def restore(self):
        for network_name in FLAGS.attack_networks:
            network.restore(self.sess, network_name)


if __name__ == '__main__':
    sess = tf.Session()

    model = Attacker(sess)
    df = PNGDataFlow(FLAGS.img_dir, FLAGS.test_list_filename, FLAGS.ground_truth_file, img_num=FLAGS.img_num)
    df = BatchData(df, FLAGS.batch_size)
    # df = PrefetchDataZMQ(df)
    df.reset_state()

    total_batch = df.ds.img_num / FLAGS.batch_size
    for batch_index, (x_batch, y_batch, name_batch) in tqdm(enumerate(df), total=total_batch):
        advs = model.perturb(x_batch, y_batch)
        save_images(advs, name_batch, FLAGS.result_dir)  # TODO: optimize this line
