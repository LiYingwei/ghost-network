import argparse

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import entropy


def eval_once(P_path, Q_path):
    # print(P_path)
    # print(Q_path)
    P = np.load(P_path)
    Q = np.load(Q_path)
    JSD = 0
    assert len(P) == 50000
    assert len(Q) == 50000
    for p, q in zip(P, Q):
        m = (p + q) / 2
        JSD += (entropy(p, m) + entropy(q, m)) / 2
    print(P_path, Q_path, JSD)
    JSD /= P.shape[0]
    return JSD


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument("--P", type=str)
    # parser.add_argument("--Q", type=str)
    # args = parser.parse_args()

    P_PATHS = ['resnet_v2_50', 'resnet_v2_50_official', 'resnet_v2_50_38', 'resnet_v2_50_205',
               'resnet_v2_50_fix_0.200-A', 'resnet_v2_50_fix_0.200-B', 'resnet_v2_50_fix_0.200-F',
               'resnet_v2_50_0.200-A']
    Q_PATHS = ['resnet_v2_50', 'resnet_v2_50_official', 'resnet_v2_50_38', 'resnet_v2_50_205',
               'resnet_v2_50_fix_0.200-A', 'resnet_v2_50_fix_0.200-B', 'resnet_v2_50_fix_0.200-F',
               'resnet_v2_50_0.200-A']
    rst = np.array(Parallel(n_jobs=36)(delayed(eval_once)('softmax_result/' + p_path + '.npy',
                                                 'softmax_result/' + q_path + '.npy')
                              for p_path in P_PATHS for q_path in Q_PATHS))\
        .reshape((len(P_PATHS), len(Q_PATHS)))
    print rst

    print(",".join(Q_PATHS))
    for p in rst:
        p_str = ["{:.4f}".format(tmp) for tmp in p]
        print(",".join(p_str))
