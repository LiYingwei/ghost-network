import os

from config import config as FLAGS
FLAGS.random_range = 0.0
from networks import network
from utils import *

accs = []
for network_name in FLAGS.test_network:
    sess = tf.Session()
    print("evaluating {:s}...".format(network_name))
    x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
    y_input = tf.placeholder(tf.int64, shape=None)

    test_img_name = load_data(FLAGS.test_list_filename)
    logits, preds = network.model(sess, x_input, network_name)

    correct_num = 0.
    dataset_meta = DatasetMetadata(FLAGS.ground_truth_file)
    for image_index, img in enumerate(test_img_name):
        orig = load_image(os.path.join(FLAGS.result_dir, img))
        label = sess.run(preds, {x_input: orig.reshape((1,) + orig.shape)})[0]
        true_label = dataset_meta.get_true_label(img[:-4] + '.pkl')
        correct_num += int(label == true_label)

    acc = correct_num / len(test_img_name)
    print("{:s}: {:.2f}%".format(network_name, 100 - acc * 100))
    accs.append(1 - acc)

    tf.reset_default_graph()
    network._network_initialized = False
    sess.close()

ndprint(FLAGS.test_network, "{:s}")
ndprint(np.array(accs) * 100)
