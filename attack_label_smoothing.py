import os

from config import config as FLAGS
from networks import network
from utils import *


class Model:
    def __init__(self, sess):
        self.x = x = tf.placeholder(tf.float32, (299, 299, 3))
        self.y = y = tf.placeholder(tf.int32)
        y_one_hot = tf.one_hot(y, 1001) * FLAGS.smoothing_factor + (1-FLAGS.smoothing_factor) / 1001
        x_expanded = tf.expand_dims(x, axis=0)
        ensemble_xs = tf.concat([x_expanded for _ in range(FLAGS.ensemble_num)], axis=0)
        ensemble_logits, ensemble_preds = network.model(sess, ensemble_xs, FLAGS.attack_network)
        ensemble_labels = tf.tile(tf.expand_dims(y_one_hot, axis=0), (ensemble_logits.shape[0], 1))
        ensemble_loss = tf.reduce_mean(
            tf.nn.softmax_cross_entropy_with_logits(logits=ensemble_logits, labels=ensemble_labels))
        self.grad, = tf.gradients(ensemble_loss, x)
        self.sess = sess

    def perturb(self, x_nat, y):
        """Given a set of examples (x_nat, y), returns a set of adversarial
           examples within epsilon of x_nat in l_infinity norm."""
        # x = x_nat + np.random.uniform(-FLAGS.max_epsilon, FLAGS.max_epsilon, x_nat.shape)
        x = np.copy(x_nat)
        lower_bound = np.clip(x_nat - FLAGS.max_epsilon, 0, 1)
        upper_bound = np.clip(x_nat + FLAGS.max_epsilon, 0, 1)

        for i in range(FLAGS.num_steps):
            grad = self.sess.run(self.grad, feed_dict={self.x: x, self.y: y})
            x = np.add(x, FLAGS.step_size * np.sign(grad), out=x, casting='unsafe')
            x = np.clip(x, lower_bound, upper_bound)

        return x


x_batches = load_data(FLAGS.test_list_filename)
sess = tf.Session()

model = Model(sess)

dataset_meta = DatasetMetadata(FLAGS.ground_truth_file)
for image_index, img in enumerate(x_batches):
    y = dataset_meta.get_true_label(img[:-4] + '.pkl')
    orig = load_image(os.path.join(FLAGS.test_img_dir, img))
    adv = model.perturb(orig, y)
    save_image(os.path.join(FLAGS.result_dir, img), adv)

    if image_index % FLAGS.report_step == 0:
        print('{} images have been processed'.format(image_index))
