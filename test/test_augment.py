import cv2 as cv
import matplotlib.pyplot as plt
from utils.preprocess import *

fig, axes = plt.subplots(2, 2)
filename = 'data/sample/demo.jpg'
img = plt.imread(filename)
img = (img / 255.).astype(np.float32)
for ax in axes.flatten():
    im = adjust_hsv(img).numpy()
    ax.imshow(im)

plt.show()