from config import config as FLAGS
from utils import *


def check(result_dir=FLAGS.result_dir, origin_dir=FLAGS.test_img_dir):
    print("checking {:s}...".format(result_dir))

    xs = load_data(FLAGS.test_list_filename)
    ys = get_label(xs, FLAGS.ground_truth_file)
    x_batches = split_to_batches(xs, FLAGS.batch_size)
    y_batches = split_to_batches(ys, FLAGS.batch_size)
    for batch_index, (x_batch, y_batch) in enumerate(zip(x_batches, y_batches)):
        advs = load_images(x_batch, result_dir)
        oris = load_images(x_batch, origin_dir)
        l_inf = np.amax(np.abs(advs - oris))
        assert l_inf < FLAGS.max_epsilon + 1e-5


if __name__ == '__main__':
    FLAGS.result_dir = 'archived/result'
    if FLAGS.result_dir == 'archived/result':
        for dir in next(os.walk(FLAGS.result_dir))[1]:
            check(os.path.join(FLAGS.result_dir, dir))
    else:
        check()
