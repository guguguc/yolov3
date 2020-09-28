import argparse

import tensorflow as tf
from tensorflow import keras

from main.detector import Detector
from utils.parser import get_config_def

"""usage
python convert_weight.py ckpt_path out_path
"""

parser = argparse.ArgumentParser()
parser.add_argument('net', action='store',
                    help='net cfg file.')
parser.add_argument('checkpoint', action='store',
                    help='checkpoint file to convert.')
parser.add_argument('out', action='store',
                    help='model file store path.')
args = vars(parser.parse_args())

net_cfg_path = args.get('net')
ckpt_path = args.get('checkpoint')
output_path = args.get('out')

model_def = get_config_def(net_cfg_path)
model = Detector(model_def, weight=ckpt_path)

