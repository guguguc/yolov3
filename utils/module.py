import tensorflow as tf
import logging
from utils.parser import *
from tensorflow import keras
from utils.common import Conv, Dense, YoloHead


LAYER = [
    'convolutional', 'dense', 'maxpool',
    'dropout', 'yolo', 'shortcut',
    'route', 'upsample'
]


def get_config_def(filename) -> list:
    parser = NetParser(filename)
    return parser.parse()


def build_model(model_params: list) -> (keras.Model, list):
    logger = logging.getLogger('main.moudle')
    net_info = [
        model_params.pop(0) for item in model_params[:]
        if item['type'] not in LAYER
    ]
    inputs = keras.Input(shape=(416, 416, 3), dtype=tf.float32, batch_size=16)
    layer_list = []
    outs = []
    y = None
    first_dense = False
    msg = ''
    yolo_index = 1
    #yolo_head = []
    yolo_losses = []
    logger.info('start to build net')
    for idx, layer_param in enumerate(model_params):
        layer_type = layer_param['type']
        if layer_type == 'convolutional':
            filters = layer_param['filters']
            size = layer_param['size']
            strides = layer_param['stride']
            padding = layer_param.get('padding', 0)
            activation = layer_param['activation']
            pad = layer_param.get('pad', 0)
            bn = layer_param.get('batch_normalize', False)
            if pad:
                padding = 'same'
            l = Conv(filters, size, strides, padding,
                     activation=activation, bn=bn,
                     name=f'conv_{idx:03d}')
            y = l(inputs) if idx == 0 else l(y)

        elif layer_type == 'maxpool':
            size = layer_param['size']
            strides = layer_param['stride']
            l = keras.layers.MaxPool2D(size, strides,
                                       padding='same')
            y = l(y)

        elif layer_type == 'dropout':
            rate = layer_param['rate']
            l = keras.layers.Dropout(rate, name=f'dropout_{idx}')
            y = l(y)

        elif layer_type == 'dense':
            if not first_dense:
                layer_list.append(keras.layers.Flatten())
                first_dense = True
            size = layer_param['size']
            activation = layer_param['activation']
            bn = layer_param['batchnormlization']
            l = Dense(size, activation, bn, name=f'dense_{idx}')
            y = l(y)

        elif layer_type == 'yolo':
            grid_size = y.shape[1]
            layer_param.update(name=f'yolo_{idx}', grid_size=grid_size)
            l1 = parse_yolo(layer_param)
            l2 = YoloHead(anchors=l1.match_anchors, name=f'yolo-{yolo_index}')
            # add the previous layer to the final out
            outs.append(l2(y))
            yolo_losses.append(l1)
            #yolo_head.append(l2)
            yolo_index += 1

        elif layer_type == 'shortcut':
            activation = 'relu'
            src = layer_param.get('from')
            l = keras.layers.Add()
            func = keras.activations.get(activation)
            y = func(l([y, layer_list[src]]))

        elif layer_type == 'route':
            src = layer_param.get('layers')
            groups = layer_param.get('groups', 1)
            group_id = layer_param.get('group_id', 0)
            if isinstance(src, int):
                src = [src]
            l = keras.layers.Concatenate(axis=-1)
            if groups == 1 and group_id == 0:
                if len(src) != 1:
                    y = l([layer_list[i] for i in src])
                else:
                    y = layer_list[src[0]]
            else:
                y = tf.split(layer_list[src[0]], num_or_size_splits=groups, axis=-1)[group_id]
            msg += f' route use {groups} group, group id is {group_id}\n'

        elif layer_type == 'upsample':
            l = keras.layers.UpSampling2D(size=layer_param['stride'],
                                          interpolation='bilinear')
            y = l(y)

        else:
            raise ValueError(f'unrecognized layer type {layer_type} ')

        msg += f' [*] layer {idx}, layer name {layer_type}, out shape: {y.shape}\n'
        layer_list.append(y)
    logger.info(msg)
    model = keras.Model(inputs=inputs, outputs=outs, name="YOLO")
    model.summary()
    keras.utils.plot_model(model, to_file='data/moudle.png',
                           show_shapes=True, show_layer_names=True,
                           dpi=150)
    return model, yolo_losses
