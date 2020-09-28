import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from glob import glob
from tensorflow import keras

policy = keras.experimental.CosineDecay(initial_learning_rate=0.002,
                                        decay_steps=2000)

warmup_steps = 100
total_steps = 1000
lr_init = 1e-3
lr_end = 1e-6
ans = []
for i in tf.range(total_steps):
    if i < warmup_steps:
        lr = (i / warmup_steps) * lr_init
    else:
        # lr = lr_end + 0.5 * (lr_init - lr_end) \
        #      * (1 + tf.cos((i - warmup_steps) / (total_steps - warmup_steps) * np.pi))
        lr = keras.experimental.CosineDecay(initial_learning_rate=lr_init,
                                            decay_steps=900)(i - 100)
    ans.append(lr)
plt.plot(ans)
plt.show()

# a = tf.random.normal(shape=[16, 10, 3])
# b = tf.random.normal(shape=[16, 15, 3])
# c = tf.TensorArray(dtype=tf.float32, size=2, dynamic_size=True)
# c.write(0, a)
# c.write(1, b)

img_path = 'data/dataset/test/'
files = glob(img_path + '*.jpg')
with open('data/test.txt', mode='w+') as fp:
    for file in files:
        path = str(Path(file).absolute().resolve())
        fp.write(path + '\n')
