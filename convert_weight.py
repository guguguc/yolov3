import argparse
import tensorflow as tf
from main.detector import Detector
from utils.module import get_config_def

"""usage
python convert_weight.py ckpt_path out_path
"""

parser = argparse.ArgumentParser()
parser.add_argument('net', action='store',
                    help='net cfg file.')
parser.add_argument('checkpoint', action='store',
                    help='checkpoint file to convert.')
parser.add_argument('savepath', action='store',
                    help='model file store path.')

args = parser.parse_args()
net_cfg_path = args.net
ckpt_path = args.checkpoint
save_path = args.savepath

model_def = get_config_def(net_cfg_path)
model = Detector(model_def, weight=ckpt_path)
inputs = tf.keras.Input((416, 416, 3), dtype=tf.float32)
out = model(inputs)
call_out = model.call.get_concrete_function(tf.TensorSpec([None, 416, 416, 3], tf.float32))
tf.saved_model.save(model,
                    save_path,
                    signatures={'serving_default': call_out})

