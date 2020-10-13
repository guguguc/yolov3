import math

import numpy as np
import pandas as pd
import tensorflow as tf


def calc_iou_xyxy(boxes_1, boxes_2, xywh=False):
    """
    :arg boxes_1, boxes_2 can be 1->N, N->1, M->N
    :param boxes_1: (a, b, ..., 4)
    :param boxes_2: (A, b, ..., 4)
    :return a:A = 1:1 1:n n:1 n:n
    """
    boxes_1_wh = boxes_1[..., 2:] - boxes_1[..., :2]
    boxes_2_wh = boxes_2[..., 2:] - boxes_2[..., :2]
    boxes_1_area = boxes_1_wh[..., 0] * boxes_1_wh[..., 1]
    boxes_2_area = boxes_2_wh[..., 0] * boxes_2_wh[..., 1]

    left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
    right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])
    intersection = tf.maximum(right_down - left_up, 0)
    intersection = intersection[..., 0] * intersection[..., 1]
    union = boxes_1_area + boxes_2_area - intersection
    iou = tf.math.divide_no_nan(intersection, union)
    return iou


def calc_iou_xywh(boxes_1, boxes_2):
    """
    :arg boxes_1, boxes_2 can be 1->N, N->1, M->N
    :param boxes_1: (a, b, ..., 4)
    :param boxes_2: (A, b, ..., 4)
    :return a:A = 1:1 1:n n:1 n:n
    """
    boxes_1_area = boxes_1[..., 2] * boxes_1[..., 3]
    boxes_2_area = boxes_2[..., 2] * boxes_2[..., 3]
    boxes_1 = xywh2xyxy(boxes_1)
    boxes_2 = xywh2xyxy(boxes_2)

    left_up = tf.maximum(boxes_1[..., :2], boxes_2[..., :2])
    right_down = tf.minimum(boxes_1[..., 2:], boxes_2[..., 2:])
    intersection = tf.maximum(right_down - left_up, 0)
    intersection = intersection[..., 0] * intersection[..., 1]
    union = boxes_1_area + boxes_2_area - intersection
    iou = intersection / union
    return iou


def calc_iou_wh(wh1, wh2):
    area1 = wh1[..., 0] * wh1[..., 1]
    area2 = wh2[..., 0] * wh2[..., 1]
    overlap_wh = tf.minimum(wh1, wh2)
    overlap = overlap_wh[..., 0] * overlap_wh[..., 1]
    union = area1 + area2 - overlap
    iou = overlap / union
    return iou


def calc_iou_np(boxes_1, boxes_2):
    boxes_1_wh = boxes_1[..., 2:] - boxes_1[..., :2]
    boxes_2_wh = boxes_2[..., 2:] - boxes_2[..., :2]
    boxes_1_area = boxes_1_wh[..., 0] * boxes_1_wh[..., 1]
    boxes_2_area = boxes_2_wh[..., 0] * boxes_2_wh[..., 1]
    left_up = np.maximum(boxes_1[..., :2], boxes_2[..., :2])
    right_down = np.minimum(boxes_1[..., 2:], boxes_2[..., 2:])
    intersection = np.maximum(right_down - left_up, 0)
    intersection = intersection[..., 0] * intersection[..., 1]
    union = boxes_1_area + boxes_2_area - intersection
    iou = np.nan_to_num(intersection / union)
    return iou


def calc_iou_wh_np(wh1, wh2):
    xyxy1 = np.concatenate([np.zeros_like(wh1), wh1], axis=-1)
    xyxy2 = np.concatenate([np.zeros_like(wh2), wh2], axis=-1)
    return calc_iou_np(xyxy1, xyxy2)


def xywh2xyxy_coco(boxes):
    xy_1 = boxes[..., :2]
    xy_2 = xy_1 + boxes[..., 2:]
    xyxy = tf.concat([xy_1, xy_2], axis=-1)
    return xyxy


def xywh2yxyx(boxes):
    boxes = xywh2xyxy(boxes)
    boxes = tf.concat([boxes[..., :2][..., ::-1],
                       boxes[..., 2:][..., ::-1]],
                      axis=-1)
    return boxes


def yxyx2xywh(boxes):
    wh = boxes[..., 2:][..., ::-1] - boxes[..., :2][..., ::-1]
    xy = boxes[..., :2][..., ::-1] + 0.5 * wh
    xywh = tf.concat([xy, wh], axis=-1)
    return xywh


def xyxy2xywh(boxes):
    wh = boxes[..., 2:] - boxes[..., :2]
    xy = boxes[..., :2] + 0.5 * wh
    xywh = tf.concat([xy, wh], axis=-1)
    return xywh


def xywh2xyxy(boxes):
    xy, wh = tf.split(boxes, [2, 2], axis=-1)
    xyxy = tf.concat([xy - 0.5 * wh, xy + 0.5 * wh], axis=-1)
    return xyxy


def xywh2xyxy_np(boxes):
    xy, wh = np.split(boxes, [2], axis=-1)
    xyxy = np.concatenate([xy - 0.5 * wh, xy + 0.5 * wh], axis=-1)
    return xyxy


def nms_gpu_v2(prediction,
               iou_threshold=0.5,
               score_threshold=0.005,
               max_size_total=1000,
               max_size_per_class=500):
    """
    :param prediction: [batchsize, N, 4 + 1 + num_class]
    :param iou_threshold: 超过此阈值的预测边界框被抑制
    :param score_threshold: 去除置信度小于此阈值或类别分数小于此阈值的边界框
    :param max_size_total: 返回的最大边界框总数
    :param max_size_per_class: 每一个类别返回的最大边界框总数
    :return: nms_info: Tensor with shape of [batchsize, max_size_total, 4 + 1 + 1]
    :return: valid_indices: Tensor with shape [batchsize], 表示batch中每个样本的有效索引
    """
    conf_mask = prediction[..., 4] >= score_threshold
    pred_scores = prediction[..., 4][..., tf.newaxis] * prediction[..., 5:]
    score_mask = tf.reduce_max(pred_scores, axis=-1) >= score_threshold
    filter_mask = tf.logical_and(conf_mask, score_mask)
    filter_mask = tf.tile(filter_mask[..., tf.newaxis], [1, 1, 10])
    pred_scores = tf.where(filter_mask, pred_scores, tf.zeros_like(pred_scores))
    pred_boxes_yxyx = xywh2yxyx(prediction[..., :4])[..., tf.newaxis, :]
    nms_boxes, nms_scores, nms_cls, valid_indices = tf.image.combined_non_max_suppression(
        pred_boxes_yxyx,
        pred_scores,
        iou_threshold=iou_threshold,
        max_total_size=max_size_total,
        max_output_size_per_class=max_size_per_class,
        score_threshold=score_threshold)
    # yxyx2xyxy
    nms_boxes = tf.concat([nms_boxes[..., :2][..., ::-1],
                           nms_boxes[..., 2:][..., ::-1]],
                          axis=-1)
    nms_scores = nms_scores[..., tf.newaxis]
    nms_cls = tf.cast(nms_cls[..., tf.newaxis], tf.float32)
    nms_info = tf.concat([nms_boxes, nms_scores, nms_cls], axis=-1)
    return nms_info, valid_indices


def nms_cpu(pred_boxes,
            pred_scores,
            iou_threshold=0.5,
            scores_threshold=5e-4,
            max_keep_per_class=200):
    """
    :param pred_boxes: (N, 4)
    :param pred_scores: (N, num_classes)
    :param iou_threshold: IOU超过此阈值的边界框将被过滤
    :param scores_threshold: scores小于此阈值的被过滤
    :param max_keep_per_class: 每个类别最大输出边界框数目
    :return: filtered_boxes
    :return: filtered_scores
    :return: classes_indices
    """
    num_classes = pred_scores.shape[1]
    boxes_indices = []
    classes_indices = []
    for cls in range(num_classes):
        scores = pred_scores[..., cls]
        indices = py_nms(pred_boxes,
                         scores,
                         iou_threshold=iou_threshold,
                         scores_threshold=scores_threshold,
                         max_keep=max_keep_per_class)
        boxes_indices.extend(indices)
        classes_indices.extend([cls] * len(indices))
    filtered_boxes = pred_boxes[boxes_indices]
    filtered_scores = pred_scores[boxes_indices]
    return filtered_boxes, filtered_scores, classes_indices


def py_nms(pred_boxes,
           pred_scores,
           iou_threshold=0.5,
           scores_threshold=5e-4,
           max_keep=50):
    """
    :param pred_boxes: 预测边界框，(N, 4)
    :param pred_scores: 预测分数，(N, 1)
    :param iou_threshold: IOU阈值，超过这个阈值的边界框预测将被抑制
    :param scores_threshold: 分数阈值，低于此阈值的scores将被忽略
    :param max_keep: 过滤后剩下的最大box数目
    :return: filtered_boxes: 过滤后的boxes, (M, 4)
    :return: filtered_scores: 过滤后的scores, (M, 1)
    """
    descend_indices = np.argsort(pred_scores, axis=-1)[::-1]
    mask = pred_scores[descend_indices] >= scores_threshold
    descend_indices = descend_indices[mask]
    keep_indices = []
    while descend_indices.size > 0:
        # 分数高，添加到keep indices
        idx = descend_indices[0]
        keep_indices.append(idx)
        ious = calc_iou_xyxy(pred_boxes[idx],
                             pred_boxes[descend_indices[1:]])
        # 保留的索引
        remeain = np.where(ious <= iou_threshold)[0]
        # 除去descend_indices中第一个，剩余的
        descend_indices = descend_indices[remeain + 1]
    keep_indices = keep_indices[:max_keep]
    return keep_indices


def correct_box(boxes, width, height, out_size=416):
    """将letter box转化为非letter box
    :param boxes: Tensor of shape [N, 4] letter box normalized coord
    :param width: Tensor of shape [N]
    :param height: Tensor of shape [N]
    :param out_size: letterbox变换后的图像尺寸
    :return: boxes: Tensor of shape [N, 4] absolute coord
    """
    ratio = np.minimum(out_size / width, out_size / height)
    new_w = np.round(width * ratio).astype(np.int32)
    new_h = np.round(height * ratio).astype(np.int32)
    pad_w = np.floor((out_size - new_w) / 2)
    pad_h = np.floor((out_size - new_h) / 2)
    boxes = boxes * out_size
    boxes[..., [0, 2]] -= pad_w
    boxes[..., [1, 3]] -= pad_h
    boxes /= ratio
    boxes[..., [0, 1]] = np.minimum(np.maximum(boxes[..., [0, 1]], 0), width - 1.)
    boxes[..., [1, 3]] = np.minimum(np.maximum(boxes[..., [1, 3]], 0), height - 1.)
    return boxes


def calc_area_pd(df: pd.DataFrame, columns: list):
    xmin, ymin, xmax, ymax = [df[col] for col in columns]
    area = (xmax - xmin) * (ymax - ymin)
    mask = (xmax < xmin) | (ymax < ymin)
    area[mask] = 0
    return area


def calc_iou_pd(df: pd.DataFrame, columns: list):
    inter_xmin, inter_ymin, inter_xmax, inter_ymax = [df[col] for col in columns]
    inter_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin)
    print(inter_area.head(10))


def normlize_boxes(boxes, height, width):
    """归一化boxes
    :param boxes: ymin, xmin, ymax, xmax
    :param height:
    :param width:
    :return:
    """
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=1)
    xmin = xmin / width
    ymin = ymin / height
    xmax = xmax / width
    ymax = ymax / height
    return tf.concat([ymin, xmin, ymax, xmax], axis=-1)


def denormlize_boxes(boxes, height, width):
    """反归一化边界框
    :param height: 图像高度
    :param width: 图像宽度
    :param boxes: 边界框参数 [None, 4]
    :return:
    """
    height = tf.cast(height, tf.float32)
    width = tf.cast(width, tf.float32)
    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=1)
    xmin = width * xmin
    ymin = height * ymin
    xmax = width * xmax
    ymax = height * ymax
    return tf.concat([ymin, xmin, ymax, xmax], axis=1)


def clip_boxes(boxes, img_shape):
    if boxes.shape[-1] != 4:
        raise ValueError('boxes.shape should be [None, 4]!')
    if isinstance(img_shape, list) or isinstance(img_shape, tuple):
        height, width = img_shape
        max_length = [height - 1.0, width - 1.0, height - 1.0, width - 1.0]
    else:
        img_shape = tf.cast(img_shape, dtype=boxes.dtype)
        height, width = tf.unstack(img_shape, axis=-1)
        max_length = tf.stack([height - 1.0, width - 1.0, height - 1.0, width - 1.0], axis=-1)

    cliped_boxes = tf.math.maximum(tf.math.minimum(boxes, max_length), 0.0)
    return cliped_boxes


def resize_and_crop_boxes(boxes, img_scale, out_size, padding, offset):
    """
    :param boxes: 边界框 Tensor of shape [None, 4]
    :param img_scale: 放缩后未加padding时图像的高度宽度与原图的比值 (2,)
    :param out_size: 图像输出尺寸 (2,)
    :param padding: padding (2,
    :param offset: 使用random jitter时crop的左上偏移 (2,)
    :return:
    """
    boxes *= tf.tile(tf.expand_dims(img_scale, axis=0), [1, 2])
    boxes -= tf.tile(tf.expand_dims(offset, axis=0), [1, 2])
    boxes += tf.tile(tf.expand_dims(padding, axis=0), [1, 2])
    boxes = clip_boxes(boxes, out_size)
    return boxes


def caculate_padded_size(desired_size, stride):
    # (100, 200) -> 32 (128, 224)
    if isinstance(desired_size, list) or isinstance(desired_size, tuple):
        padded_size = [math.ceil(size * 1.0 / stride) * stride for size in desired_size]
        return padded_size
    else:
        return tf.cast(
            tf.math.ceil(tf.cast(desired_size, tf.float32) / stride) * stride,
            dtype=tf.int32)


def flip_left_right_boxes(boxes):
    """水平反转边界框
    :param boxes: Tensor of shape [None, 4]
    :return: 水平反转后的box
    """
    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=1)
    flip_xmin = tf.math.subtract(1.0, xmax)
    flip_xmax = tf.math.subtract(1.0, xmin)
    return tf.concat([ymin, flip_xmin, ymax, flip_xmax], axis=1)
