import logging

import tensorflow as tf
import numpy as np

from functools import partial

from tensorflow import keras

from utils.bbox import calc_iou_vectorition_xywh, calc_iou_vectorition_wh, calc_iou_vectorition_wh_np
from utils.util import filter_duplicate
from config.config import *

np.set_printoptions(precision=4, suppress=True)


# tf.debugging.set_log_device_placement(True)

class BatchNormlization(keras.layers.BatchNormalization):
    """
    make set trainable False is not to run in infer mode
    """

    def call(self, inputs, training=False):
        if not training:
            traininginputs_shape = tf.constant(False)
        training = tf.logical_and(training, self.trainable)
        return super(BatchNormlization, self).call(inputs, training)


class Conv(keras.layers.Layer):
    def __init__(self,
                 filters,
                 kernel_size,
                 strides=1,
                 padding='valid',
                 activation='relu',
                 bn=False,
                 name=None):
        super(Conv, self).__init__(name=name)
        initializer = keras.initializers.random_normal(stddev=0.01)
        self.conv = keras.layers.Conv2D(filters,
                                        kernel_size,
                                        strides,
                                        padding=padding,
                                        use_bias=not bn,
                                        kernel_initializer=initializer,
                                        kernel_regularizer=keras.regularizers.l2(5e-3))
        if bn:
            self.batchnorm = keras.layers.BatchNormalization()
        self.bn = bn
        if activation == 'leaky':
            self.activation = partial(keras.activations.relu, alpha=0.1)
        else:
            self.activation = keras.activations.get(activation)

    def call(self, x, training=False):
        x = self.conv(x)
        if self.bn:
            x = self.batchnorm(x, training=training)
        x = self.activation(x)
        return x


class Dense(keras.layers.Layer):
    def __init__(self,
                 size,
                 activation='relu',
                 bn=True,
                 name=None):
        super(Dense, self).__init__(name=name)
        self.fc = keras.layers.Dense(size, use_bias=not bn)
        if bn:
            self.bn = keras.layers.BatchNormalization()
        self.activation = keras.activations.get(activation)

    def call(self, x, training=True, **kwargs):
        for layer in self.layers:
            x = layer(x, training=training)
        x = self.activation(x)
        return x


class Route(keras.layers.Layer):
    def __init__(self,
                 src,
                 groups=1,
                 group_id=0,
                 name=None):
        super(Route, self).__init__(name=name)
        self.src = [src] if isinstance(src, int) else src
        self.multiple = len(src) != 1
        self.groups = groups
        self.group_id = group_id
        self.merge = keras.layers.Concatenate(axis=-1)


class Aggregator(keras.layers.Layer):
    def __init__(self):
        super(Aggregator, self).__init__()
        shape = [BATCH_SIZE, -1, 4 + 1 + CLASS_NUM]
        self.map_fn = partial(tf.reshape, shape=shape)
        self.merge_fn = partial(tf.concat, axis=1)

    def call(self, inputs, training=None):
        """
        inputs: [[b, s1, s1, 3, 15], [b, s2, s2, 3, 15]]
        """
        out_large = self.map_fn(inputs[0])
        out_small = self.map_fn(inputs[1])
        out = self.merge_fn([out_large, out_small])
        return out


class YoloHead(keras.layers.Layer):
    def __init__(self, anchors, name=None):
        super(YoloHead, self).__init__(name=name)
        self.anchors = anchors

    def call(self, inputs, **kwargs):
        inputs_shape = tf.shape(inputs)
        batch_size = inputs_shape[0]
        grid_size = inputs_shape[1]
        prediction = tf.reshape(
            inputs, (batch_size, grid_size, grid_size, 3, 4 + 1 + CLASS_NUM))
        raw_xy = tf.nn.sigmoid(prediction[..., 0:2])
        raw_wh = prediction[..., 2:4]
        pred_conf = tf.nn.sigmoid(prediction[..., 4:5])
        pred_cls = tf.nn.sigmoid(prediction[..., 5:])
        prediction = tf.concat([raw_xy, raw_wh, pred_conf, pred_cls], axis=-1)
        prediction = encode_outputs(prediction, grid_size, self.anchors)
        return prediction


def encode_outputs(inputs, grid_size, anchors):
    grid_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2), tf.float32)
    raw_xy, raw_wh, conf, cls = tf.split(inputs, [2, 2, 1, CLASS_NUM], axis=-1)
    grid_size = tf.cast(grid_size, tf.float32)
    anchors = tf.reshape(anchors, [1, 1, 1, -1, 2])
    pred_xy = (raw_xy + grid_xy) / grid_size
    pred_wh = tf.exp(raw_wh) * anchors
    # anchors should relative to grid not img
    outputs = tf.concat([pred_xy, pred_wh, conf, cls], axis=-1)
    return outputs


def decode_outputs(inputs, grid_size, anchors):
    grid_xy = tf.meshgrid(tf.range(grid_size), tf.range(grid_size))
    grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2), tf.float32)
    pred_xy, pred_wh, pred_conf, pred_cls = tf.split(inputs, [2, 2, 1, CLASS_NUM], axis=-1)
    grid_size = tf.cast(grid_size, tf.float32)
    anchors = tf.reshape(anchors, [1, 1, 1, -1, 2])
    raw_xy = pred_xy * grid_size - grid_xy
    raw_wh = tf.math.log(pred_wh / anchors + 1e-16)
    outputs = tf.concat([raw_xy, raw_wh, pred_conf, pred_cls], axis=-1)
    return outputs


class YOLOLoss:
    def __init__(self,
                 anchors,
                 mask,
                 num_classes,
                 grid_size,
                 img_dim=416,
                 params=None):
        self.img_dim = tf.convert_to_tensor(img_dim, dtype=tf.float32)
        self.anchors = tf.cast(tf.truediv(anchors, img_dim), tf.float32)
        self.anchors = tf.reshape(self.anchors, [-1, 2])
        self.mask = mask
        self.match_anchors = tf.gather(self.anchors, indices=mask)
        self.num_classes = num_classes

        # loss function
        self.mse_loss = tf.losses.mean_squared_error
        self.bce_loss = tf.losses.binary_crossentropy
        self.sparse_loss = tf.losses.sparse_categorical_crossentropy
        self.l1_smooth_loss = tf.losses.huber

        self.obj_scale = 1
        self.noobj_scale = 100
        self.metric = dict()

        params = params if params else dict()
        self.ignore_thresh = params.pop('ignore_thresh', 0.7)
        self.truth_thresh = params.pop('truth_thresh', 1.)
        # [iou, mse, giou, diou, ciou]
        self.iou_loss = params.pop('iou_loss', 'iou')
        # [iou, giou, diou, ciou]
        self.iou_thresh_type = params.pop('iou_thresh_kind', 'iou')
        self.iou_thresh = params.pop('iou_thresh', 1.0)
        # [default, greedynms, diounms]
        self.nms_kind = params.pop('nms_kind', 'default')
        self.beta_nms = params.pop('beta_nms', 0.6)
        self.max_delta = params.pop('max_delta', 5)
        self.iou_normalizer = params.pop('iou_normalizer', 0.07)
        self.cls_normalizer = params.pop('cls_normalizer', 1.)

        self.grid_size = tf.convert_to_tensor(grid_size, tf.float32)
        self.grid_xy = None
        self.stride = None
        self.init_grid_param()

        self.logger = logging.getLogger(f'main.yolo-{self.grid_size}')

    def init_grid_param(self):
        grid_xy = tf.meshgrid(tf.range(self.grid_size), tf.range(self.grid_size))
        self.grid_xy = tf.cast(tf.expand_dims(tf.stack(grid_xy, axis=-1), axis=2), tf.float32)
        self.grid_size = tf.cast(self.grid_size, tf.float32)
        self.stride = self.img_dim / self.grid_size
        self.stride = tf.truediv(self.img_dim, self.grid_size)

    def deocode_gt_box(self, box, indices):
        # 相对于一个网格的偏移(0-1)
        box = tf.gather_nd(box, indices[..., :2])
        anchors = tf.gather(self.match_anchors, indices[..., 2])
        xy = box[..., :2] * self.grid_size - tf.floor(box[..., :2] * self.grid_size)
        wh = tf.math.log(box[..., 2:] / anchors + 1e-16)
        box = tf.concat([xy, wh], -1)
        return box

    def __call__(self, y_pred, y_true):
        gt_box, gt_cls = tf.split(y_true, [4, 1], axis=-1)
        pred_box, pred_conf, pred_cls = tf.split(y_pred, [4, 1, self.num_classes], axis=-1)
        raw_box = decode_outputs(y_pred,
                                 grid_size=self.grid_size,
                                 anchors=self.match_anchors)[..., :4]

        # t1 = time.perf_counter()
        obj_mask, noobj_mask, cls_mask, bnyxc = self.build_target(pred_box, y_true)
        obj_count = tf.reduce_sum(tf.cast(obj_mask, tf.int32))
        noobj_count = tf.reduce_sum(tf.cast(noobj_mask, tf.int32))
        # t2 = time.perf_counter()
        # print('function build_target const {int((t2-t1)*1000)} ms')

        # 将gt box的值解码为pred raw的值
        bnc = tf.concat([bnyxc[..., :2], bnyxc[..., 4:5]], axis=-1)
        gt_box = self.deocode_gt_box(gt_box, bnc)

        pos_obj = tf.boolean_mask(pred_conf, mask=obj_mask)
        neg_obj = tf.boolean_mask(pred_conf, mask=noobj_mask)
        cls_mask = tf.boolean_mask(cls_mask, obj_mask)
        pos_xy = tf.boolean_mask(raw_box[..., :2], mask=obj_mask)
        pos_wh = tf.boolean_mask(raw_box[..., 2:], mask=obj_mask)
        val_pred_cls = tf.boolean_mask(pred_cls, obj_mask)

        loss_obj = 1.0 * self.bce_loss(y_pred=pos_obj,
                                       y_true=tf.ones(shape=[obj_count, 1], dtype=tf.float32),
                                       from_logits=False)

        loss_noobj = 1.0 * self.bce_loss(y_pred=neg_obj,
                                         y_true=tf.zeros(shape=[noobj_count, 1], dtype=tf.float32),
                                         from_logits=False)
        loss_cls = self.sparse_loss(y_pred=val_pred_cls, y_true=cls_mask)
        loss_xy = 1.0 * self.bce_loss(y_pred=pos_xy, y_true=gt_box[..., :2])
        loss_wh = 2.0 * self.l1_smooth_loss(y_pred=pos_wh, y_true=gt_box[..., 2:])

        # show info
        obj_count = tf.cast(obj_count, tf.float32)
        noobj_count = tf.cast(noobj_count, tf.float32)
        cls_accu_mask = tf.argmax(val_pred_cls, -1, tf.int32) == cls_mask
        cls_accu = tf.reduce_mean(tf.cast(cls_accu_mask, tf.float32)) * 100.
        grid_pobj = tf.reduce_sum(tf.boolean_mask(pred_conf, obj_mask)) / obj_count
        grid_pnoobj = tf.reduce_sum(tf.boolean_mask(pred_conf, noobj_mask)) / noobj_count
        grid_pxy = tf.reduce_sum(self.mse_loss(y_pred=pos_xy, y_true=gt_box[..., :2])) / obj_count
        grid_pwh = tf.reduce_sum(self.mse_loss(y_pred=pos_wh, y_true=gt_box[..., 2:])) / obj_count
        self.logger.debug(f'\ntotal obj   count {obj_count}'
                          f'\ngrid  obj   prob {grid_pobj}'
                          f'\ngrid  noobj prob {grid_pnoobj}'
                          f'\ngrid  xy    prob {grid_pxy}'
                          f'\ngrid  wh    prob {grid_pwh}'
                          f'\ngrdi  cls   accu {cls_accu}')

        loss_obj = tf.reduce_sum(loss_obj) / obj_count
        loss_noobj = 100.0 * tf.reduce_sum(loss_noobj) / noobj_count
        loss_cls = tf.reduce_sum(loss_cls) / obj_count
        loss_wh = tf.reduce_sum(loss_wh) / obj_count
        loss_xy = tf.reduce_sum(loss_xy) / obj_count
        loss_box = (loss_wh + loss_xy)
        return loss_box, loss_obj, loss_noobj, loss_cls

    def build_target(self, pred_box, label):
        """
        :param pred_box: Tensor of shape (batch, grid_size, grid_size, num_scale, 4)
        :param label: Tensor of shape (batch, num_label, 4 + 1)
        :return obj_mask: Tensor of shape (batch, g, g, num of scale), tf.bool type
        :return noobj_mask: Tensor represemt background grid with the shame shape and type of obj_mask
        """
        shape = tf.shape(pred_box)
        b, g, s = shape[0], shape[1], shape[3]
        gt_box, gt_cls = tf.split(label, [4, 1], axis=-1)
        gt_box_xy, gt_box_wh = tf.split(gt_box, [2, 2], axis=-1)
        # 计算anchor与gt wh的iou
        iou = calc_iou_vectorition_wh(self.anchors[tf.newaxis, ...], gt_box_wh[..., tf.newaxis, :])
        # 最佳匹配
        iou_best = tf.argmax(iou, axis=-1, output_type=tf.int32)
        bidx = tf.tile(tf.range(b)[..., tf.newaxis], [1, tf.shape(iou_best)[1]])
        iidx = tf.tile(tf.range(tf.shape(iou_best)[1])[tf.newaxis, ...], [b, 1])
        iou_best = tf.stack([bidx, iidx, iou_best], axis=-1)
        anchor_match_mask = tf.logical_and(iou_best[..., -1] >= self.mask[0],
                                           iou_best[..., -1] <= self.mask[2])
        # 非空边界框
        nonzerobox_mask = tf.reduce_all(tf.equal(gt_box, tf.zeros(shape=[1])), axis=-1)
        nonzerobox_mask = tf.logical_not(nonzerobox_mask)
        # nonzerobox_mask = tf.expand_dims(nonzerobox_mask, axis=-1)
        # 最匹配且不为空box
        match_mask = tf.logical_and(nonzerobox_mask, anchor_match_mask)
        iou_best = tf.boolean_mask(iou_best, match_mask)
        # iou_best = tf.logical_and(iou_best, nonzerobox)
        # 不是最佳但是超过阈值，将一个gt分配到多个同一格点的anchor中
        # iou_extra = iou > self.iou_thresh
        # iou_match = tf.logical_or(iou_best, iou_extra)
        grid_yx = tf.boolean_mask(gt_box[..., :2][..., ::-1], match_mask) * (tf.cast(g, tf.float32))
        grid_yx = tf.cast(tf.floor(grid_yx), tf.int32)
        n = iou_best[..., 1:2]
        byxs = tf.concat([iou_best[..., :1], grid_yx, iou_best[..., 2:3] - self.mask[0]], axis=-1)
        byxs, unique_idx = filter_duplicate(byxs)
        n = tf.gather(n, indices=unique_idx)
        cls_onethot = tf.cast(tf.boolean_mask(gt_cls, match_mask), tf.int32)
        cls_onethot = tf.gather(cls_onethot, indices=unique_idx)
        obj_mask = tf.scatter_nd(indices=byxs, updates=tf.ones(tf.shape(byxs)[0], tf.bool),
                                 shape=[b, g, g, s])
        cls_mask = tf.scatter_nd(indices=byxs, updates=tf.squeeze(cls_onethot, axis=-1),
                                 shape=tf.shape(obj_mask))

        # 16 13 13 3
        # boxes = tf.scatter_nd(indices=byxs, updates=gt_box, shape=[b, g, g, s, 4])
        # iou = calc_iou_vectorition_xywh(pred_box, boxes)
        iou = self.caculate_iou_in_batch(pred_box, gt_box, match_mask)
        self.ignore_thresh = 0.5
        ignore_mask = iou > self.ignore_thresh
        # 除去最佳匹配的
        noobj_mask = tf.logical_not(tf.logical_or(ignore_mask, obj_mask))
        bnyxsc = tf.gather(indices=[0, 4, 1, 2, 3, 5],
                           params=tf.concat([byxs, n, cls_onethot], axis=-1),
                           axis=-1)
        return obj_mask, noobj_mask, ignore_mask, cls_mask, bnyxsc

    def caculate_iou_in_batch(self, pred_box, gt_box, val_mask):
        """计算每个batch中所有预测格点与gt的IOU"""

        def func(box1, box2):
            box1 = tf.expand_dims(box1, axis=-2)
            box2 = tf.expand_dims(box2, axis=0)
            shape = tf.broadcast_dynamic_shape(tf.shape(box1), tf.shape(box2))
            box1 = tf.broadcast_to(box1, shape)
            box2 = tf.broadcast_to(box2, shape)
            return calc_iou_vectorition_xywh(box1, box2)

        # 计算所有gt与所有预测框的iou
        # bach size 20 4 and bach size 13 13 3 4
        # 1 20 4  13 13 3 1 4
        # 13 13 3 1 4 and 1 20 4
        # 13 13 3
        ious = tf.map_fn(fn=lambda x: tf.reduce_max(func(x[0], tf.boolean_mask(x[1], x[2])),
                                                    axis=-1),
                         elems=[pred_box, gt_box, val_mask],
                         fn_output_signature=tf.TensorSpec(
                             shape=[None, None, 3], dtype=tf.float32))
        return ious
