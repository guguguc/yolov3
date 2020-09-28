import tensorflow as tf
from functools import partial
from tensorflow import keras


device = tf.config.list_physical_devices('GPU')[0]
tf.config.experimental.set_memory_growth(device, True)
x = tf.random.normal([1, 416, 416, 3])
kernal = tf.random.normal([1, 3, 3, 3])
initialier = keras.initializers.RandomNormal(0.0, 1.0, seed=1)
conv1 = keras.layers.Conv2D(1,
                            3,
                            2,
                            padding='same',
                            kernel_initializer=initialier)
conv2 = keras.layers.Conv2D(1,
                            3,
                            2,
                            padding='valid',
                            kernel_initializer=initialier)
y1 = conv1(x)
x = keras.layers.ZeroPadding2D(((0, 1), (0, 1)))(x)
y2 = conv2(x)
print(y2==y1)
