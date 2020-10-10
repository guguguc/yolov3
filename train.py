from main.trainer import Trainer
from main.detector import Detector
from utils.module import get_config_def

# weight_file = 'data/weights/yolov4-tiny.conv.29'
# weight_file = 'data/weights/yolov3-tiny.conv.15'
# weight_file = 'data/weights/yolov4-tiny_best.weights'
weight_file = 'data/weights/yolov3-tiny.weights'
cut_off = ['conv_015', 'conv_022']

# config = 'config/yolov4-tiny.cfg'
config = 'config/yolov3-tiny.cfg'
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
# trainer.profile()
# mean_ap, ap = trainer.eval()
# print(mean_ap)
