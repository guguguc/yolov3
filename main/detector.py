import logging
import os

import numpy as np
import tensorflow as tf
from terminaltables import AsciiTable

from config.config import CLASS_NUM
from utils.build import build_model

DEBUG = False
if DEBUG:
    from utils.util import debugging

    debugging('logs/debugging')


class Detector(tf.keras.Model):
    def __init__(self,
                 net_params,
                 weight=None,
                 cut_off=None):
        """
        net_params  [list]: net config params.
        weight       [str]: weight file name.
        cut_of [int, list]: layer id of cut.
        """
        super().__init__()
        self.backbone, self.yolo_losses = build_model(net_params)
        self.num_head = len(self.yolo_losses)
        self.logger = logging.getLogger(__name__)
        if weight:
            self.load_weight(weight, cut_off)

    @tf.function
    def call(self, img):
        """predict func.
        do once net forward and combine all ouput.
        """
        outputs = self.backbone(img, training=False)
        batch_size = tf.shape(outputs[0])[0]
        shape = (batch_size, -1, 4 + 1 + CLASS_NUM)
        outputs = [tf.reshape(out, shape) for out in outputs]
        outputs = tf.concat(outputs, axis=1)
        return outputs

    def forward(self, img, label, training=None):
        """loss aggregate func.
        do once net forward and sum all out loss.
        """
        outputs = self.backbone(img, training=training)
        loss_box, loss_obj, loss_noobj, loss_cls = 0, 0, 0, 0
        for out, loss_func in zip(outputs, self.yolo_losses):
            lbox, lobj, lnoobj, lcls = loss_func(out, label)
            loss_box += lbox
            loss_obj += lobj
            loss_noobj += lnoobj
            loss_cls += lcls
        return loss_box, loss_obj, loss_noobj, loss_cls

    def load_weight(self, path, cut_off=None):
        """load net weight
        support tensorflow checkpoint or darknet weight.
        """
        prefix = path.split('.')[0]
        if prefix.startswith('ckpt'):
            self.load_ckpt(path)
        else:
            self.load_darknet(path, cut_off)

    def load_ckpt(self, ckpt_path):
        self.logger.info(f'started to load ckpt in {ckpt_path}')
        ckpt = tf.train.Checkpoint(self.backbone)
        ckpt.restore(ckpt_path).expect_partial()

    def load_darknet(self, weight_path, cut_off=None):
        self.logger.info('started to load weight')
        size = os.path.getsize(weight_path)
        self.logger.info(f'total size {size} bytes, {size / (1 << 20):.2f}MB')
        fp = open(weight_path, 'rb')
        # training info
        major, minor, revision, seen, _ = np.fromfile(fp, dtype=np.int32, count=5)
        layer_list = [layer.name for layer in self.backbone.layers if layer.name[:4] == 'conv']
        layer_list.sort()
        # load partical layers
        if cut_off:
            if isinstance(cut_off, int):
                cut_off_index = layer_list.index(f'conv_{cut_off:03d}')
                layer_list = layer_list[:cut_off_index]
            if isinstance(cut_off, list):
                layer_list = [name for name in layer_list if name not in cut_off]
        info_msg = ''
        # used for caculate bytes num
        previous_read = fp.tell()
        for idx, layer_name in enumerate(layer_list):
            layer = self.backbone.get_layer(layer_name)
            info_msg += f' [*] {layer_name} {layer.output_shape}\n'
            load_weights = []
            weights = layer.weights
            kernel = layer.weights[0]
            filters = kernel.shape[-1]
            # load bn layer
            if len(weights) == 5:
                # gamma, beta, moving_mean, moving_variance = tuple(weights[1:])
                bn_weight = np.fromfile(fp, dtype=np.float32, count=4 * filters)
                bn_weight = bn_weight.reshape([4, filters])[[1, 0, 2, 3]]
                load_weights.extend([w for w in bn_weight])
            # load conv bias
            elif len(weights) == 2:
                bias_weight = np.fromfile(fp, dtype=np.float32, count=filters)
                load_weights.append(bias_weight)
            previous_read = fp.tell()
            # load conv kernal
            # darknet       conv shape [out_dim, in_dim, height, width]
            # tensorflow    conv shape [height, width, in_dim, out_dim]
            darknet_shape = np.array(kernel.shape)[[3, 2, 0, 1]]
            weight_kernel = np.fromfile(fp, dtype=np.float32, count=np.product(kernel.shape))
            weight_kernel = weight_kernel.reshape(darknet_shape).transpose(2, 3, 1, 0)
            # combine all weights of this layer
            load_weights.insert(0, weight_kernel)
            layer.set_weights(load_weights)
            previous_read = fp.tell()
            self.logger.debug(f'layer {layer.name}, kernel shape {kernel.shape}')
            self.logger.debug(f'read increnment: {fp.tell() - previous_read}, '
                              f'total read {fp.tell()} bytes, '
                              f'{fp.tell() / (1 << 20):.2f} MB')
        assert len(fp.read()) == 0, 'falied to load weight'
        self.logger.info(info_msg[5:])
        fp.close()

    def debug(self):
        """
        show loss info of every out featuremap.
        """
        metrics = self.yolo_losses[0].metrics
        metrics_list = [loss.get_metric() for loss in self.yolo_losses]
        cols = [[m] for m in metrics]
        for i, var in enumerate(metrics_list):
            for j, m in enumerate(metrics):
                cols[j].append(f'{var[m]:.3f}')
        table = AsciiTable(table_data=cols)
        print('\n' + table.table)
        for loss in self.yolo_losses:
            loss.reset_metric()
