import numpy as np
import tensorflow as tf


def optimistic_restore(session, save_file):
    reader = tf.train.NewCheckpointReader(save_file)
    saved_shapes = reader.get_variable_to_shape_map()
    var_names = sorted([(var.name, var.name.split(':')[0]) for var in tf.global_variables()
                        if var.name.split(':')[0] in saved_shapes])
    restore_vars = []
    with tf.variable_scope('', reuse=True):
        for var_name, saved_var_name in var_names:
            curr_var = tf.get_variable(saved_var_name)
            var_shape = curr_var.get_shape().as_list()
            if var_shape == saved_shapes[saved_var_name]:
                restore_vars.append(curr_var)
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)


def load_data(test_list_filename):
    with open(test_list_filename, 'r') as f:
        test_img_name = f.readlines()
        test_img_name = [x.strip() for x in test_img_name]

    return test_img_name


def load_image(path):
    return np.load(path) / 255.0


def save_image(path, image):
    np.save(path, image * 255)


def ndprint(a, format_string='{:2.2f}%, '):
    str = ''
    for v in a:
        str += format_string.format(v * 100)
    print(str)

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

    def get_true_label(self, image_id):
        """Returns true label for image with given ID."""
        return self._true_labels[image_id]
