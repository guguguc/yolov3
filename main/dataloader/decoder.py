import tensorflow as tf


class TfExampleDecoder:
    def __init__(self):
        self.features = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'img_path': tf.io.FixedLenFeature([], tf.string),
            'img': tf.io.FixedLenFeature([], tf.string),
            'xmin': tf.io.VarLenFeature(tf.float32),
            'ymin': tf.io.VarLenFeature(tf.float32),
            'xmax': tf.io.VarLenFeature(tf.float32),
            'ymax': tf.io.VarLenFeature(tf.float32),
            'obj_id': tf.io.VarLenFeature(tf.int64),
            'obj_name': tf.io.VarLenFeature(tf.string)
        }

    @staticmethod
    def _decode_img(raw_img):
        img = tf.io.decode_jpeg(raw_img, channels=3)
        img.set_shape([None, None, 3])
        return img

    @staticmethod
    def _decode_boxes(xmin, ymin, xmax, ymax):
        return tf.stack([ymin, xmin, ymax, xmax], axis=-1)

    def decode(self, serialized_example):
        """解码序列化tf example
        :param serialized_example:
        :return:
            decoded_tensor: 解码后的tensor字典
              - height: int64 tensor 图像高度
              - width: int64 tensor 图像宽度
              - img: uint8 tensor 解码后图像内容 [None, None, 3]
              - gt_boxes: float32 tensor 根据高度与宽度归一化的边界框信息 [None, 4]
              - gt_classes: int64 tensor 目标类别 [None]
        """
        example = tf.io.parse_single_example(serialized_example, features=self.features)

        # 稀疏元素转换为dense
        for k in example:
            if isinstance(example[k], tf.SparseTensor):
                if example[k].dtype == tf.string:
                    example[k] = tf.sparse.to_dense(example[k], default_value='')
                else:
                    example[k] = tf.sparse.to_dense(example[k], default_value=0)

        raw_img = example['img']
        img_id = tf.strings.substr(example['img_path'], pos=-16, len=12)
        xmin, ymin = example['xmin'], example['ymin']
        xmax, ymax = example['xmax'], example['ymax']
        img = self._decode_img(raw_img)
        boxes = self._decode_boxes(xmin, ymin, xmax, ymax)
        decoded_tensor = {
            'height': example['height'],
            'width': example['width'],
            'img': img,
            'img_id': img_id,
            'gt_boxes': boxes,
            'gt_classes': tf.cast(example['obj_id'], tf.float32) - 1.
        }
        return decoded_tensor
