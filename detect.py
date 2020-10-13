import argparse
import glob

import cv2 as cv
import numpy as np
import tensorflow as tf

from config.config import CLASS_NAME, IMG_SIZE
from utils.bbox import nms_gpu_v2, correct_box
from utils.image import resize_and_crop_image
from utils.util import enable_mem_group
from utils.visual import draw_boxes

enable_mem_group()

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


def detect_img(img, infer_func):
    cv.cvtColor(img, cv.COLOR_BGR2RGB, dst=img)
    h, w, c = img.shape
    out_size = (IMG_SIZE, IMG_SIZE)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img, info = resize_and_crop_image(img, out_size, out_size)
    img = tf.expand_dims(img, 0)
    out = infer_func(img)['output_0']
    nms_info, val_idx = nms_gpu_v2(out,
                                   args.iou_threshold, args.conf_thresh,
                                   args.max_detct_num)
    nms_info = nms_info[0][:val_idx[0]].numpy()
    box, score, cls = np.split(nms_info, [4, 5], axis=-1)
    # convert letterbox coord to origin img coord
    w, h, c = img.shape
    box = correct_box(box, w, h, IMG_SIZE)
    cls = cls.astype(np.int)
    return box, score, cls


def detect_video(video, infer_func):
    cap = cv.VideoCapture(video)
    while cap.isOpened():
        ret, frame = cap.read()
        box, score, cls = detect_img(frame, infer_func)
        draw_boxes(frame, box, cls=cls, classes=CLASS_NAME, thickness=2)
        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    args = parser.parse_args()
    print(args)
    loaded_model = tf.saved_model.load(args.weight)
    infer = loaded_model.signatures['serving_default']
    file_lists = glob.glob(args.file + '/*')
    for file in file_lists:
        img = cv.imread(file)
        box, score, cls = detect_img(img, infer)
        if not box.size:
            continue
        img = draw_boxes(img, box, cls=cls, classes=CLASS_NAME, thickness=2)
        cv.imwrite(args.file + '/out/', img)
