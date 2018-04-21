import os

import numpy as np
import tensorflow as tf

from networks.core import resnet_v2_152 as network

# test_list_filename = '../data/FlorianProject/test_shortlist_100.txt'
# label_list_filename = '../data/FlorianProject/label_shortlist_100.txt'
test_list_filename = '../data/FlorianProject/test_list.txt'
label_list_filename = '../data/FlorianProject/label_list.txt'
#  load image list
with open(test_list_filename, 'rb') as f:
    test_img_name = f.readlines()

test_img_dir = '../data/FlorianProject/test_data/'
test_img_name = [x.strip() for x in test_img_name]

sess = tf.Session()

#  model for attack / generate robust images
x_input = tf.placeholder(tf.float32, (None, 299, 299, 3))
y_input = tf.placeholder(tf.int64, shape=None)
logits, preds = network.model(sess, x_input)

labels = []
for img_index, img in enumerate(test_img_name):
    #  load image
    orig = np.load(os.path.join(test_img_dir, img))
    orig /= 255.0
    label = sess.run(preds, {x_input: orig.reshape((1,) + orig.shape)})
    labels.append("{}\n".format(label[0]))

with open(label_list_filename, "w") as f:
    f.writelines(labels)

