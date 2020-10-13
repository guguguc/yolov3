import tensorflow as tf


class TfExampleEncoder:
    def __init__(self):
        self.features = {
            'height': None,
            'width': None,
            'img_path': None,
            'img': None,
            'xmin': None,
            'ymin': None,
            'xmax': None,
            'ymax': None,
            'obj_name': None,
            'obj_id': None}

    def encode(self, img_path: str, boxes: list, obj_id: list, obj_name: list):
        feature = self.features
        img_raw = open(img_path, 'rb').read()
        img_shape = tf.image.decode_image(img_raw, channels=3)
        height, width, _, = img_shape
        xmin = boxes[:, 0] / width
        ymin = boxes[:, 1] / height
        xmax = boxes[:, 2] / width
        ymax = boxes[:, 3] / height
        img_path = img_path.encode('utf8')
        obj_name = [name.encode('utf8') for name in obj_name]
        feature['height'] = tf.train.Feature(Int64List=tf.train.Int64List([], value=[height]))
        feature['width'] = tf.train.Feature(Int64List=tf.train.Int64List([], value=[width]))
        feature['img_path'] = tf.train.Feature(BytesList=tf.train.BytesList([], value=[img_path]))
        feature['img'] = tf.train.Feature(BytesList=tf.train.BytesList([], value=[img_raw]))
        feature['xmin'] = tf.train.Feature(FloatList=tf.train.FloatList([], value=[xmin]))
        feature['ymin'] = tf.train.Feature(FloatList=tf.train.FloatList([], value=[ymin]))
        feature['xmax'] = tf.train.Feature(FloatList=tf.train.FloatList([], value=[xmax]))
        feature['ymax'] = tf.train.Feature(FloatList=tf.train.FloatList([], value=[ymax]))
        feature['obj_name'] = tf.train.Feature(BytesList=tf.train.BytesList([], value=[obj_name]))
        feature['obj_id'] = tf.train.Feature(Int64List=tf.train.Int64List([], value=[obj_id]))
        return tf.train.Example(features=tf.train.Features(feature=feature)).SerializeToString()
