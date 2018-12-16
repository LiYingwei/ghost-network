import argparse
import os

import numpy as np
import tensorflow as tf
from scipy.misc import imread, toimage

parser = argparse.ArgumentParser(description='convert npy to png')
parser.add_argument('--input_dir', type=str, default='/home/yingwei/lyw/data/val_data')
parser.add_argument('--output_dir', type=str, default='/home/yingwei/lyw/data/val_data_png')
args = parser.parse_args()

if not os.path.exists(args.output_dir):
    os.mkdir(args.output_dir)

for filepath in tf.gfile.Glob(os.path.join(args.input_dir, '*.npy')):
    image = np.load(filepath)
    basename = os.path.basename(filepath)
    base, ext = os.path.splitext(basename)

    assert ext == ".npy"
    savepath = os.path.join(args.output_dir, base + ".png")
    toimage(image, cmin=0, cmax=255).save(savepath)
    image_loaded = imread(savepath, mode='RGB').astype(np.float)
    # import pdb; pdb.set_trace()
    assert np.sum(image - image_loaded) == 0, "{:s},{:f}".format(filepath, np.sum(image - image_loaded))
