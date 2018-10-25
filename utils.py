import os

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
            else:
                print(var_shape, saved_shapes[saved_var_name])
    saver = tf.train.Saver(restore_vars)
    saver.restore(session, save_file)

    # fix_names = [var for var in tf.global_variables()
    #              if 'fix_weight/weight' in var.name or 'fix_dropout/random' in var.name]
    # for var in fix_names:
    #     var.initializer.run(session=session)



def load_data(test_list_filename):
    with open(test_list_filename, 'r') as f:
        test_img_name = f.readlines()
        test_img_name = [x.strip() for x in test_img_name]

    return test_img_name


def load_data_with_checking(test_list_filename, result_dir):
    with open(test_list_filename, 'r') as f:
        test_img_name = f.readlines()
        test_img_name = [x.strip() for x in test_img_name if not os.path.exists(os.path.join(result_dir, x.strip()))]

    return test_img_name


def load_image(path):
    return np.load(path) / 255.0


def load_images(filenames, dir):
    return np.stack([load_image(os.path.join(dir, filename)) for filename in filenames])


def save_image(path, image):
    np.save(path, image * 255)


def save_images(advs, x_batch, dir):
    for adv, x in zip(advs, x_batch):
        save_image(os.path.join(dir, x), adv)


def ndprint(a, format_string='{:2.2f}%, '):
    str = ''
    for v in a:
        str += format_string.format(v)
    print(str)


def ndstr(a, format_string='{:2.2f}%, '):
    str = ''
    for v in a:
        str += format_string.format(v)
    return str


def split_to_batches(xs, batch_size):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(xs), batch_size):
        yield xs[i:i + batch_size]


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


def get_label(xs, ground_truth_file):
    dataset_meta = DatasetMetadata(ground_truth_file)
    return dataset_meta.get_true_label([x[:-4] + '.pkl' for x in xs])
