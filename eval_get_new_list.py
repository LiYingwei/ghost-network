from config import config as FLAGS
from networks import network
from utils import *

accs = []
FLAGS.test_network = ["resnet_v2_101", "resnet_v2_50"]
all_network_labels = []

xs = load_data(FLAGS.test_list_filename)
ys = get_label(xs, FLAGS.ground_truth_file)

for network_name in FLAGS.test_network:
    sess = tf.Session()
    print("evaluating {:s}...".format(network_name))
    x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
    _, preds = network.model(sess, x_input, network_name)

    all_labels = []
    x_batches = split_to_batches(xs, FLAGS.batch_size)
    y_batches = split_to_batches(ys, FLAGS.batch_size)
    for batch_index, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
        images = load_images(x_batch, FLAGS.test_img_dir)
        labels = sess.run(preds, {x_input: images})
        all_labels.append(labels)
    print(len(all_labels))
    all_network_labels.append(np.concatenate(all_labels, axis=0))

    tf.reset_default_graph()
    network._network_initialized = False
    sess.close()

correct_index = np.logical_and(all_network_labels[0] == ys, all_network_labels[1] == ys)
# print(np.array(xs)[correct_index])

with open("resnet_testlist.txt", "w") as f:
    for x in np.array(xs)[correct_index]:
        f.writelines(x + '\n')
