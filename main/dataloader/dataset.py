from main.dataloader.decoder import TfExampleDecoder
from utils.bbox import yxyx2xywh, normlize_boxes, denormlize_boxes, resize_and_crop_boxes, caculate_padded_size
from utils.image import *
from utils.util import *

enable_mem_group()


class Transformer:
    def __init__(self, params, mode):
        self.out_size = params.get('output_size', (416, 416))
        self.max_level = params.get('max_level', 5)
        self.aug_random_flip = params.get('random_filp', False)
        self.aug_scale_min = params.get('scale_min', 1.0)
        self.aug_scale_max = params.get('scale_max', 1.0)
        self.max_num_instance = params.get('max_num_instance', 100)
        self.mode = mode
        self.is_training = mode == 'train'

        self.example_decoder = TfExampleDecoder()
        if mode == 'train':
            self.parse_fn = self.parse_train_data
        elif mode == 'val':
            self.parse_fn = self.parse_eval_data

    def __call__(self, value):
        with tf.name_scope('data_transform'):
            data = self.example_decoder.decode(value)
            data = self.parse_fn(data)
            return data

    def parse_train_data(self, data):
        height, width = data['height'], data['width']
        gt_boxes, gt_classes = data['gt_boxes'], data['gt_classes']
        img_id = data['img_id']
        img = normalize_img(data['img'])
        # t1 = tf.timestamp()
        img, gt_boxes = random_horizontal_filp(img, gt_boxes)
        # t2 = tf.timestamp()
        # tf.print((t2-t1)*1000.0)
        gt_boxes = denormlize_boxes(gt_boxes, height, width)
        img, img_info = resize_and_crop_image(img, self.out_size,
                                              self.out_size,
                                              self.aug_scale_min, self.aug_scale_max)
        # resize and crop boxes
        padding = img_info[0, :]
        desire_size = img_info[1, :]
        img_scale = img_info[2, :]
        offset = img_info[3, :]
        gt_boxes = resize_and_crop_boxes(gt_boxes, img_scale, desire_size, padding, offset)
        gt_boxes = normlize_boxes(gt_boxes, self.out_size[0], self.out_size[1])
        gt_boxes = yxyx2xywh(gt_boxes)
        gt_classes = tf.reshape(gt_classes, [-1, 1])
        label = tf.concat([gt_boxes, gt_classes], -1)
        return img, img_id, label

    def parse_eval_data(self, data):
        height, width = data['height'], data['width']
        gt_boxes, gt_classes = data['gt_boxes'], data['gt_classes']
        img = normalize_img(data['img'])
        img_id = data['img_id']
        gt_boxes = denormlize_boxes(gt_boxes, height, width)
        img, img_info = resize_and_crop_image(img, self.out_size,
                                              caculate_padded_size(self.out_size, 2 ** self.max_level),
                                              aug_scale_min=1.0, aug_scale_max=1.0)
        padding = img_info[0, :]
        desire_size = img_info[1, :]
        img_scale = img_info[2, :]
        offset = img_info[3, :]
        gt_boxes = resize_and_crop_boxes(gt_boxes, img_scale, desire_size, padding, offset)
        gt_boxes = normlize_boxes(gt_boxes, self.out_size[0], self.out_size[1])
        gt_boxes = yxyx2xywh(gt_boxes)
        gt_classes = tf.reshape(gt_classes, [-1, 1])
        label = tf.concat([gt_boxes, gt_classes], axis=-1)
        return img, img_id, label


class DataSet:
    def __init__(self,
                 batch_size: int,
                 tf_record_path: str,
                 mode: str,
                 params: dict,
                 shuffle_buffer=512):
        self.batch_size = batch_size
        self.is_training = mode == 'train'
        self.imgs_num = 1000
        self.shuffle_buffer = shuffle_buffer
        self.epoch = params.get('epoch', 50)
        self.file_pattern = str(Path(tf_record_path).absolute().resolve()) + '/*'
        self.parse_fn = Transformer(params, mode)
        self.data_fn = tf.data.TFRecordDataset
        self.mode = mode

    def __call__(self, batch_size: int = None):
        if not batch_size:
            batch_size = self.batch_size
        dataset = tf.data.Dataset.list_files(self.file_pattern)
        dataset = dataset.interleave(
            map_func=self.data_fn, cycle_length=2,
            num_parallel_calls=2,
            deterministic=True
        )
        dataset = dataset.map(self.parse_fn,
                              num_parallel_calls=tf.data.experimental.AUTOTUNE)
        # if self.is_training:
        #    dataset = dataset.shuffle(self.shuffle_buffer)
        padded_shapes = ([None, None, 3], [], [None, 5])
        padded_values = (0., "", 0.)

        dataset = dataset.padded_batch(batch_size=batch_size,
                                       padded_shapes=padded_shapes,
                                       padding_values=padded_values,
                                       drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset
