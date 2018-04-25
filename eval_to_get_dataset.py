from config import config as FLAGS
from networks import network
from utils import *

val_img_dir = '/home/cihang/Research/adv_mit/data_mitigate/val_data/'


def get_list_in_dir(dir):
    import glob, os
    return map(os.path.basename, glob.glob(os.path.join(dir, "*.npy")))


FLAGS.test_network = ["resnet_v2_50", "resnet_v2_101"]

xs = get_list_in_dir(val_img_dir)
ys = get_label(xs, FLAGS.ground_truth_file)

for network_name in FLAGS.test_network:
    sess = tf.Session()
    print("evaluating {:s}...".format(network_name))
    x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
    _, preds = network.model(sess, x_input, network_name)

    all_labels = []
    x_batches = split_to_batches(xs, FLAGS.batch_size)
    for batch_index, x_batch in enumerate(x_batches):
        images = load_images(x_batch, val_img_dir)
        labels = sess.run(preds, {x_input: images})
        all_labels.append(labels)
    network_labels = np.concatenate(all_labels, axis=0)

    with open('imagnet_val_' + network_name + '.csv', "w") as f:
        for x, y in zip(xs, network_labels):
            f.writelines("{:s},{:d}\n".format(x, y))

    tf.reset_default_graph()
    network._network_initialized[network_name] = False
    sess.close()
