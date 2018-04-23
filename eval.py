import os

from config import config as FLAGS
from networks import network
from utils import *

accs = []
for network_name in FLAGS.test_network:
    sess = tf.Session()
    print("evaluating {:s}...".format(network_name))
    x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
    _, preds = network.model(sess, x_input, network_name)

    correct_num = 0.
    xs = load_data(FLAGS.test_list_filename)
    ys = get_label(xs, FLAGS.ground_truth_file)
    x_batches = split_to_batches(xs, FLAGS.batch_size)
    y_batches = split_to_batches(ys, FLAGS.batch_size)
    for batch_index, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
        images = load_images(x_batch, FLAGS.result_dir)
        gt_labels = y_batch
        labels = sess.run(preds, {x_input: images})
        correct_num += np.sum(labels == gt_labels)

    acc = correct_num / len(xs)
    print("{:s}: {:.2f}%".format(network_name, 100 - acc * 100))
    accs.append(1 - acc)

    tf.reset_default_graph()
    network._network_initialized = False
    sess.close()

ndprint(FLAGS.test_network, "{:s}, ")
ndprint(np.array(accs) * 100)
