import os

import numpy as np
import scipy
import tensorflow as tf
from scipy.misc import imread
from tensorpack import DataFlow


class DatasetMetadata(object):
    """Helper class which loads and stores dataset metadata."""

    def __init__(self, filename):
        import csv
        """Initializes instance of DatasetMetadata."""
        self._true_labels = {}
        with open(filename) as f:
            reader = csv.reader(f)
            header_row = next(reader)
            try:
                row_idx_image_id = header_row.index('name')
                row_idx_true_label = header_row.index('label')
            except ValueError:
                raise IOError('Invalid format of dataset metadata.')
            for row in reader:
                if len(row) < len(header_row):
                    # skip partial or empty lines
                    continue
                try:
                    image_id = row[row_idx_image_id]
                    self._true_labels[image_id] = int(row[row_idx_true_label])
                except (IndexError, ValueError):
                    raise IOError('Invalid format of dataset metadata')

    def get_true_label(self, image_ids):
        """Returns true label for image with given ID."""
        return [self._true_labels[image_id] for image_id in image_ids]


class PNGDataFlow(DataFlow):
    def __init__(self, imagedir, imagelistfile, gtfile, img_num=-1):
        self.imagedir = imagedir
        with open(imagelistfile, 'r') as f:
            self.imagename = f.readlines()
            self.imagename = [x.strip() for x in self.imagename]
        self.gt_dict = DatasetMetadata(gtfile)._true_labels
        self.img_num = img_num if img_num > -1 else len(self.imagename)

    def __iter__(self):
        for imgname in self.imagename[:self.img_num]:
            with tf.gfile.Open(os.path.join(self.imagedir, imgname + ".png"), 'rb') as f:
                image = imread(f, mode='RGB').astype(np.float) / 255.0
            # image = np.load(os.path.join(self.imagedir, imgname + ".npy")) / 255.0
            yield [image, self.gt_dict[imgname], imgname]


def save_images(images, savenames, savedir):
    for image, savename in zip(images, savenames):
        # np.save(os.path.join(savedir, savename + ".npy"), image * 255)
        scipy.misc.toimage(image * 255, cmin=0, cmax=255).save(os.path.join(savedir, savename + ".png"))
