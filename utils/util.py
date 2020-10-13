import glob
import os
import shutil

import numpy as np
import tensorflow as tf


def check_file_exist(file_path):
    if not isinstance(file_path, list):
        file_path = [file_path]
    for file in file_path:
        if not os.path.exists(file):
            raise FileNotFoundError(file)


def copy_files(src, dst):
    files = glob.glob(src)
    for file in files:
        shutil.copy2(file, dst)


def write_file(src, content):
    with open(src, mode='w+') as fp:
        fp.write(content)


def calc_receive_fild(net_params: list):
    previous_receivr_filed = 0
    prod_step = 1
    current_receive_fild = 0
    # r(i+1) = r(i) + size(i+1) * prod_step(i)
    for parm in net_params:
        layer_name = parm['type']
        if layer_name in ['conv', 'pool']:
            size = parm['size']
            strides = parm['strides']
            if current_receive_fild == 0:
                current_receive_fild = size
            else:
                current_receive_fild = previous_receivr_filed + (size - 1) * prod_step
            prod_step *= strides
            print(current_receive_fild)
            previous_receivr_filed = current_receive_fild

    return current_receive_fild


def calc_out_shape(input_shape, padding, size, strides):
    if isinstance(input_shape, int):
        return (input_shape + 2 * padding - size) // strides + 1
    else:
        h, w = input_shape
        out_h = (h + 2 * padding - size) // strides + 1
        out_w = (w + 2 * padding - size) // strides + 1
        return out_h, out_w


def filter_duplicate(sequence):
    # maped_array = contor_pairing_n(sequence)
    compress_array = 17 * sequence[..., 0] \
                     + 19 * sequence[..., 1] \
                     + 21 * sequence[..., 2] \
                     + 23 * sequence[..., 3]
    count = tf.range(tf.shape(compress_array)[0])
    unique, idx = tf.unique(compress_array)
    unique_first = tf.math.unsorted_segment_min(data=count,
                                                segment_ids=idx,
                                                num_segments=tf.shape(unique)[0])
    sequence = tf.gather(sequence, indices=unique_first)
    return sequence, unique_first


def calc_ap(y_pred, y_truth, gt_classs_num, num_class):
    ap = np.zeros([num_class])
    num_detections = y_pred.shape[0]
    for cls in range(num_class):
        tp = y_truth == cls
        tp_fp = y_pred == cls
        tp_accum = np.cumsum(tp)
        tp_fp_accum = np.cumsum(tp_fp)
        nonzero_bound = np.maximum(num_detections - np.count_nonzero(tp_accum),
                                   num_detections - np.count_nonzero(tp_fp_accum))
        tp_accum = tp_accum[nonzero_bound:]
        tp_fp_accum = tp_fp_accum[nonzero_bound:]
        precision = tp_accum / tp_fp_accum
        recall = tp_accum / gt_classs_num[cls]
        ap[cls] = auc_n_point_interpolation(recall, precision)
    return ap


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


def enable_mem_group():
    gpu = tf.config.list_physical_devices('GPU')[0]
    tf.config.experimental.set_memory_growth(gpu, True)


def debugging(root_path):
    tf.debugging.experimental.enable_dump_debug_info(
        dump_root=root_path,
        tensor_debug_mode='FULL_HEALTH',
        circular_buffer_size=-1)
