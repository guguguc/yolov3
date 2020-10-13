from collections import OrderedDict

from utils.blas import isfloat, isint, isseq
from utils.layer import YOLOLoss
from utils.util import check_file_exist


class Parser:
    def __init__(self, filename: str):
        check_file_exist(filename)
        self.filename = filename

    def parse(self):
        pass

    def get_lines(self, comment='#'):
        return [line.strip() for line in open(self.filename, errors='ignore') if
                line.strip() and not line.startswith(comment)]

    @staticmethod
    def ret_key_val_pair(line):
        pair_list = line.partition('=')
        if '' not in pair_list:
            key = pair_list[0].strip()
            val = pair_list[2].strip()
            val = Parser.change_val_type(val)
            return key, val
        return False

    @staticmethod
    def change_val_type(val):
        """根据key-val pair中val的格式产生不同类型的数据
        :param val: val
        :return: int, float or list
        """
        if isint(val):
            return int(val)
        elif isfloat(val):
            return float(val)
        elif isseq(val):
            seq = [v.strip() for v in val.split(',')]
            seq = map(Parser.change_val_type, seq)
            return list(seq)
        else:
            return val


class NetParser(Parser):
    """解析网络配置文件
    :param self.options 包含网络层参数字典的列表
    :param self.options['layer'] 层次名称 [net, convolutional, route, maxpool, upsample, yolo]
    """

    def __init__(self, filename: str):
        super(NetParser, self).__init__(filename)
        self.lines = self.get_lines()
        self.options = []

    def parse(self) -> list:
        for line in self.lines:
            if line.startswith('[') and line.endswith(']'):
                self.options.append(dict(type=line[1:-1]))
            else:
                pair = NetParser.ret_key_val_pair(line)
                if pair:
                    layer_key, layer_val = pair
                    self.options[-1][layer_key] = layer_val
                else:
                    raise ValueError(f'error: check your format in line {line}')
        return self.options


class ConfigParser(Parser):
    def __init__(self, filename: str):
        super(ConfigParser, self).__init__(filename)
        self.lines = self.get_lines()
        self.options = OrderedDict()

    def parse(self) -> OrderedDict:
        last_key = ''
        for line in self.lines:
            if line.startswith('[') and line.endswith(']'):
                last_key = line[1: -1]
                self.options[last_key] = dict()
            else:
                pair = NetParser.ret_key_val_pair(line)
                if pair:
                    layer_key, layer_val = pair
                    self.options[last_key][layer_key] = layer_val
                else:
                    raise ValueError(f'error: check your format in line {line}')
        return self.options


class DataParser(Parser):
    def __init__(self, filename: str):
        super(DataParser, self).__init__(filename)
        self.img_config = filename
        self.label_config = self.img_config[:-4] + '.labels'
        self.config = dict()

    def parse(self):
        """
        :return: 包含训练集，验证集图像名称与标签列表的字典
            - train -> dict(file_list: [...], labels: [img1[box1[5],box2[5], ...], img2[box1[5], box2[5]], ...]
            - val   -> dict(file_list: [...], labels: [img1[box1[5],box2[5], ...], img2[box1[5], box2[5]], ...]
        """
        img_path_list = self.get_lines()
        self.filename = self.label_config
        label_config = DataParser.split_labels(self.get_lines())
        return img_path_list, label_config

    @staticmethod
    def split_labels(img_labels_str_list: list):
        """ 将每行表示图像box参数的字符串列表划分为分割的数值列表 """
        all_labels = []
        for img_labels_str in img_labels_str_list:
            data_points = list(map(int, img_labels_str.split()))
            label = []
            for idx in range(0, len(data_points), 5):
                label.append(data_points[idx:idx + 5])
            all_labels.append(label)
        return all_labels


def parse_yolo(params: dict):
    anchors = params.pop('anchors')
    mask = params.pop('mask')
    classes = params.pop('classes', 10)
    grid_size = params.pop('grid_size')
    layer = YOLOLoss(anchors, mask, classes, grid_size, params=params)
    return layer
