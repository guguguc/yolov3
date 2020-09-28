import time
from main.dataloader.dataset import DataSet
from utils.parser import ConfigParser, Parser

config_path = 'config/tune.cfg'
config = ConfigParser(config_path).parse().pop('data')
names_path = config.pop('names')
config_path = config.pop('val')
class_names = Parser(names_path).get_lines()
record_path = 'data/tfrecord/train'
ds = DataSet(batch_size=16,
             tf_record_path=record_path,
             mode='train',
             params={})()
t = 0
for idx, data in enumerate(ds):
    current = time.perf_counter()
    internal = (current - t) * 1000
    t = current
    print(internal)
