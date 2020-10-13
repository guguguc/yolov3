import os

from main.detector import Detector
from main.trainer import Trainer
from utils.build import get_config_def

config = 'config/yolov3-tiny.cfg'
weight = ['yolov3-tiny.conv.29',
          'yolov4-tiny.conv.29',
          'yolov4-tiny-best.weights']
cut_off = ['conv_015', 'conv_022']
path = 'data/weights'

weight_file = os.path.join(path, weight[0])
params = get_config_def(config)
data_params = params.pop(0)
train_params = params.pop(0)
optimizer_params = params.pop(0)
train_params.update(optimizer=optimizer_params)
net = Detector(params,
               weight=weight_file,
               cut_off=cut_off)
restore = 'data/checkpoint/ckpt-?'
train_params['restore'] = None
trainer = Trainer(net, train_params, data_params)
trainer.train()
# mean_ap, ap = trainer.eval()
# print(mean_ap)
