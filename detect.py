import time
import glob
import argparse

import cv2 as cv
import numpy as np
import tensorflow as tf

from utils.tfconfig import enable_mem_group
from utils.bbox import nms_gpu_v2, correct_box
from utils.image import draw_boxes
from utils.preprocess import resize_and_crop_image
from config.config import CLASS_NAME

enable_mem_group()
# usage
# python detect.py yolo.cfg
#                  weight.weights
#                  -conf_thresh 0.25
#                  -max_detct_num 100
#                  [-video, -image]
#                  file[*.mp4, *.jpg]

parser = argparse.ArgumentParser(description='Yolo object detector.')
parser.add_argument('net', action='store', type=str,
                    help='config file of the object detector.')
parser.add_argument('weight', action='store', type=str,
                    help='train weight file of the object detector.')
parser.add_argument('--conf_thresh', action='store', default=0.4, type=int,
                    help='confidence threshold of determining if having a object.')
parser.add_argument('--max_detct_num', action='store', default=100, type=int,
                    help='maxiumn number of object in onece detection.')
parser.add_argument('--type', choices=['video', 'img'], type=str,
                    help='The type of file to detect.')
parser.add_argument('--file', type=str,
                    help='input file name.')
cmd = 'yolo.cfg config/yolov4_tiny.cfg video demo.jpg'

args_dict = parser.parse_args()
print(args_dict)

weight = args_dict.weight
loaded_model = tf.saved_model.load(weight)
infer = loaded_model.signatures['serving_default']
inputs_file = glob.glob(args_dict.file + '/*')
for file in inputs_file:
    img_ori = cv.imread(file)
    cv.cvtColor(img_ori, cv.COLOR_BGR2RGB, dst=img_ori)
    h, w, c = img_ori.shape
    # img = cv.resize(img, (416, 416))
    out_size = (416, 416)
    img = tf.image.convert_image_dtype(img_ori, tf.float32)
    img, info = resize_and_crop_image(img, out_size, out_size)
    img = tf.expand_dims(img, 0)
    t1 = time.perf_counter()
    out = infer(img)['output_0']
    nms_info, val_idx = nms_gpu_v2(out, 0.5, args_dict.conf_thresh, 50, 20)
    val_indices = val_idx[0].numpy()
    nms_info = nms_info[0].numpy()[:val_indices]
    nms_box, nms_score, nms_cls = np.split(nms_info,
                                           [4, 5],
                                           axis=-1)
    if nms_box.size == 0:
        continue
    nms_cls = np.squeeze(nms_cls, axis=-1).astype(np.int)
    nms_box = correct_box(nms_box, w, h, 416)
    t2 = time.perf_counter()
    internal = int((t2 - t1) * 1000)
    print(f'cost {internal} ms')
    cv.cvtColor(img_ori, cv.COLOR_RGB2BGR, dst=img_ori)
    im = draw_boxes(img_ori, nms_box, cls=nms_cls, classes=CLASS_NAME, thickness=2)
    cv.imshow('demo', im)
    cv.waitKey(0)
