import time

from cleverhans.attacks_tf import CarliniWagnerL2

from config import config as FLAGS
from networks import network
from utils import *


def get_one_hot(targets, nb_classes):
    targets = np.array(targets)
    l = len(targets)
    res = np.zeros((l, nb_classes), dtype=np.float32)
    res[np.arange(l), targets] = 1
    return res
    # res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    # return res.reshape(list(targets.shape) + [nb_classes])


class Model:
    def __init__(self, sess):
        self.sess = sess

    def get_logits(self, x_input):
        logits, _ = network.model(self.sess, x_input, FLAGS.attack_network)
        return logits


class Attacker(object):
    def __init__(self, sess):
        model = Model(sess)
        self.cw2 = CarliniWagnerL2(sess, model, FLAGS.batch_size, 10, False, 0.01, 3, 250,
                                   True, 100, 0.0, 1.0, 1001, [299, 299, 3])

    def perturb(self, imgs, labels):
        return self.cw2.attack_batch(imgs, get_one_hot(labels, 1001))


if __name__ == '__main__':
    xs = load_data_with_checking(FLAGS.test_list_filename, FLAGS.result_dir) if FLAGS.cont else load_data(
        FLAGS.test_list_filename)
    ys = get_label(xs, FLAGS.ground_truth_file)
    x_batches = split_to_batches(xs, FLAGS.batch_size)
    y_batches = split_to_batches(ys, FLAGS.batch_size)
    sess = tf.Session()

    attacker = Attacker(sess)

    start = time.time()
    for batch_index, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
        images = load_images(x_batch, FLAGS.test_img_dir)
        labels = y_batch
        advs = attacker.perturb(images, labels)
        save_images(advs, x_batch, FLAGS.result_dir)

        image_index = batch_index * FLAGS.batch_size
        if image_index % 50 == 0:
            time_used = time.time() - start
            time_predict = time_used / (batch_index + 1) * (5000 / FLAGS.batch_size - batch_index - 1)
            print('{} images have been processed, {:.2}h used, {:.2}h need'.
                  format(image_index, time_used / 3600, time_predict / 3600))
