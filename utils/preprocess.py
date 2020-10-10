import tensorflow as tf
import glob
import numpy as np
import math
from tqdm import tqdm
from pathlib import Path


def adjust_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    hsv = tf.image.rgb_to_hsv(img)
    hyp = [hgain, sgain, vgain]
    r = tf.random.uniform([3], -1, 1) * hyp + 1
    hsv = tf.clip_by_value(hsv * r, 0., 1.)
    img = tf.image.hsv_to_rgb(hsv)
    return img


def flip_left_right_boxes(boxes):
    """水平反转边界框
    :param boxes: Tensor of shape [None, 4]
    :return: 水平反转后的box
    """
    ymin, xmin, ymax, xmax = tf.split(boxes, 4, axis=1)
    flip_xmin = tf.math.subtract(1.0, xmax)
    flip_xmax = tf.math.subtract(1.0, xmin)
    return tf.concat([ymin, flip_xmin, ymax, flip_xmax], axis=1)


def random_horizontal_filp(img, boxes):
    do_flip = tf.greater(tf.random.uniform([]), 0.5)
    img = tf.cond(pred=do_flip,
                  true_fn=lambda: img,
                  false_fn=lambda: tf.image.flip_left_right(img))
    boxes = tf.cond(pred=do_flip,
                    true_fn=lambda: boxes,
                    false_fn=lambda: flip_left_right_boxes(boxes))
    return img, boxes


def normalize_img(image,
                  offset=(0.453, 0.451, 0.429),
                  scale=(0.290, 0.285, 0.298)):
    image = tf.image.convert_image_dtype(image, tf.float32)
    # offset = tf.constant(offset)
    # 保证rank一致
    # offset = tf.expand_dims(offset, axis=0)
    # offset = tf.expand_dims(offset, axis=0)
    # image -= offset

    # scale = tf.constant(scale)
    # scale = tf.expand_dims(scale, axis=0)
    # scale = tf.expand_dims(scale, axis=0)
    # image /= scale
    return image


def resize_and_crop_image(img,
                          desired_size,
                          padding_size,
                          aug_scale_min=1.0,
                          aug_scale_max=1.0,
                          seed=None,
                          method=tf.image.ResizeMethod.BILINEAR):
    img_shape = tf.cast(tf.shape(img)[:2], tf.float32)
    random_jiter = (aug_scale_min != 1.0 or aug_scale_max != 1.0)
    if random_jiter:
        random_scale = tf.random.uniform([], aug_scale_min, aug_scale_max)
        # tf.print(f'random size is {random_scale}')
        scale_size = tf.round(random_scale * desired_size)
    else:
        scale_size = desired_size
    scale_ratio = tf.minimum(scale_size[0] / img_shape[0], scale_size[1] / img_shape[1])
    scale_size = tf.round(scale_ratio * img_shape)

    img_scale = scale_size / img_shape
    # tf.print(img_scale)
    if random_jiter:
        max_offset = scale_size - desired_size
        max_offset = tf.where(tf.less(max_offset, 0),
                              tf.zeros_like(max_offset),
                              max_offset)
        offset = max_offset * tf.random.uniform([2], 0, 1, seed=seed)
        offset = tf.cast(offset, tf.int32)
        # tf.print(f'crop offset {offset.numpy()}, scale_size: {scale_size}')
    else:
        offset = tf.zeros((2,), tf.int32)
    scaled_img = tf.image.resize(img, tf.cast(scale_size, tf.int32), method=method)

    if random_jiter:
        scaled_img = scaled_img[offset[0]: offset[0] + desired_size[0], offset[1]: offset[1] + desired_size[1], :]

    scale_size = tf.cast(scale_size, tf.float32)
    padding_height = tf.maximum(tf.cast(tf.floor((desired_size[0] - scale_size[0]) / 2), tf.int32), 0)
    padding_width = tf.maximum(tf.cast(tf.floor((desired_size[1] - scale_size[1]) / 2), tf.int32), 0)
    out_img = tf.image.pad_to_bounding_box(scaled_img, padding_height, padding_width, padding_size[0], padding_size[1])
    padding = tf.cast(tf.stack([padding_height, padding_width]), tf.float32)
    img_info = tf.stack([padding,
                         tf.cast(desired_size, tf.float32),
                         img_scale,
                         tf.cast(offset, tf.float32)])
    return out_img, img_info


def caculate_mean_and_std(filepath, img_size=416):
    path = str(Path(filepath).absolute().resolve())
    files = glob.glob(path + '/*.jpg')
    imgs = tf.Variable(np.zeros(shape=(img_size, img_size, 3)), dtype=tf.float32)
    bar = tqdm(files)
    mean = tf.constant([0.45361553, 0.4518901, 0.42876361])
    mean = tf.expand_dims(mean, axis=0)
    mean = tf.expand_dims(mean, axis=0)
    for idx, file in enumerate(bar):
        im = tf.io.read_file(file)
        im = tf.io.decode_jpeg(im, channels=3)
        im.set_shape([None, None, 3])
        im = tf.image.convert_image_dtype(im, dtype=tf.float32)
        im = tf.image.resize_with_crop_or_pad(im, img_size, img_size)
        imgs.assign_add(tf.square(im - mean))
    imgs = tf.math.divide(imgs, len(files))
    std = tf.math.reduce_mean(tf.math.sqrt(imgs), axis=(0, 1))
    return mean, std


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
