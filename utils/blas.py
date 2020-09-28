import re
import numpy as np
import tensorflow as tf


def isfloat(val: str):
    try:
        float(val)
        return True
    except ValueError:
        return False


def isint(val: str):
    reg = re.compile(r'^[-+]?[0-9]+$')
    if reg.match(val):
        return True
    else:
        return False


def isseq(val: str):
    seq_list = val.split(',')
    if '' not in seq_list and len(seq_list) != 1:
        return True
    else:
        return False


def contor_pairing_n(sequence):
    k1 = sequence[..., :2]
    k2 = sequence[..., 2:]
    k1 = contor_pairing(k1)
    k2 = contor_pairing(k2)
    k1 = tf.expand_dims(k1, axis=-1)
    k2 = tf.expand_dims(k2, axis=-1)
    k = tf.concat([k1, k2], axis=-1)
    k = contor_pairing(k)
    return k


def contor_pairing(pair):
    k1 = pair[:, 0] + pair[:, 1]
    k2 = k1 + 1
    k3 = pair[:, 0]
    k = tf.math.floordiv(k1 * k2, 2) + k3
    return k


def auc_n_point_interpolation(x, y, n=101):
    # 求离散曲线下面积
    # 假定x属于[0,1]
    # 从[0-1]选择等距的n个点
    # auc_area = sum((1/(n-1)) * max(y_{i * 1/(n-1)}, y_{xmax})
    segment = np.linspace(0, 1, n)
    auc_area = 0
    for p in segment:
        mask = x >= p
        if mask.any():
            auc_area += np.max(y[mask])
    auc_area /= n
    return auc_area
