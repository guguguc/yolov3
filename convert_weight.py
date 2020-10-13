import argparse

import tensorflow as tf

from config.config import IMG_SIZE
from main.detector import Detector
from utils.build import get_config_def

"""usage
python convert_weight.py ckpt_path out_path
"""

parser = argparse.ArgumentParser()
parser.add_argument('net', action='store', help='net cfg file.')
parser.add_argument('checkpoint', action='store', help='checkpoint file to convert.')
parser.add_argument('savepath', action='store', help='model file store path.')

if __name__ == "__main__":
    args = parser.parse_args()
    model_def = get_config_def(args.net)
    DTYPE = tf.float32

    inputs = tf.keras.Input((IMG_SIZE, IMG_SIZE, 3), dtype=DTYPE)
    model = Detector(model_def, weight=args.checkpoint)
    out = model(inputs)
    out_type = tf.TensorSpec([None, IMG_SIZE, IMG_SIZE, 3], DTYPE)
    call_out = model.call.get_concrete_function(out_type)
    tf.saved_model.save(model,
                        args.savepath,
                        signatures={'serving_default': call_out})
