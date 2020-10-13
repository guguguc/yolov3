import glob
from pathlib import Path

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from utils.bbox import flip_left_right_boxes


def adjust_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
    hsv = tf.image.rgb_to_hsv(img)
    hyp = [hgain, sgain, vgain]
    r = tf.random.uniform([3], -1, 1) * hyp + 1
    hsv = tf.clip_by_value(hsv * r, 0., 1.)
    img = tf.image.hsv_to_rgb(hsv)
    return img


def random_horizontal_filp(img, boxes):
    do_flip = tf.greater(tf.random.uniform([]), 0.5)
    img = tf.cond(pred=do_flip,
                  true_fn=lambda: img,
                  false_fn=lambda: tf.image.flip_left_right(img))
    boxes = tf.cond(pred=do_flip,
                    true_fn=lambda: boxes,
                    false_fn=lambda: flip_left_right_boxes(boxes))
    return img, boxes


def normalize_img(image):
    image = tf.image.convert_image_dtype(image, tf.float32)
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


