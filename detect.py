import argparse
import tensorflow as tf
import cv2 as cv

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
parser.add_argument('-conf_thresh', action='store', default=0.25, type=int,
                    help='confidence threshold of determining if having a object.')
parser.add_argument('-max_detct_num', action='store', default=100, type=int,
                    help='maxiumn number of object in onece detection.')
parser.add_argument('type', choices=['video', 'img'], type=str,
                    help='The type of file to detect.')
parser.add_argument('file', type=str,
                    help='input file name.')
cmd = 'yolo.cfg config/yolov4_tiny.cfg video demo.jpg'

args_dict = vars(parser.parse_args())

weight = args_dict.get('weight')
loaded_model = tf.saved_model.load(weight)
infer = loaded_model.signatures['serving_default']

