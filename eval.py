import numpy as np
import tensorflow as tf
from tensorpack import (BatchData)
from tqdm import tqdm

from config import config as FLAGS
from data import PNGDataFlow
from networks import network


class Evaluator:
    def __init__(self, sess):
        self.sess = sess
        # Prepare graph
        self.build_graph()
        self.restore()

    def build_graph(self):
        batch_shape = [FLAGS.batch_size, 299, 299, 3]
        self.x_input = tf.placeholder(tf.float32, shape=batch_shape)
        self.y_input = tf.placeholder(tf.int64, shape=batch_shape[0])
        self.acc_list = []
        for network_name in FLAGS.test_networks:
            acc = network.model(self.x_input, network_name, label=self.y_input)
            self.acc_list.append(acc)

    def eval(self, images, labels):
        accs = self.sess.run(self.acc_list, feed_dict={self.x_input: images, self.y_input: labels})
        # try below lines if OOM
        # accs = []
        # for acc_tensor in self.acc_list:
        #     accs.append(self.sess.run(acc_tensor, feed_dict={self.x_input: images, self.y_input: labels}))
        return np.array(accs)

    def restore(self):
        for network_name in FLAGS.test_networks:
            network.restore(self.sess, network_name)


class AvgMetric(object):
    def __init__(self, datashape):
        self.cnt = np.zeros(datashape)
        self.sum = 0.

    def update(self, sum, cnt=1):
        self.sum += sum
        self.cnt += cnt

    def get_status(self):
        return self.sum / self.cnt


if __name__ == '__main__':
    sess = tf.Session()

    model = Evaluator(sess)
    df = PNGDataFlow(FLAGS.result_dir, FLAGS.test_list_filename, FLAGS.ground_truth_file, img_num=FLAGS.img_num)
    df = BatchData(df, FLAGS.batch_size)
    # df = PrefetchDataZMQ(df)
    df.reset_state()

    avgMetric = AvgMetric(datashape=[len(FLAGS.test_networks)])
    total_batch = df.ds.img_num / FLAGS.batch_size
    for batch_index, (x_batch, y_batch, name_batch) in tqdm(enumerate(df), total=total_batch):
        acc = model.eval(x_batch, y_batch)
        # import pdb; pdb.set_trace()
        avgMetric.update(acc)
    print(FLAGS.test_networks)
    print(np.array2string(1 - avgMetric.get_status(), separator=', ', precision=4))
