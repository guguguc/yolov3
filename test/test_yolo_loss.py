import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

from main.dataloader.dataset import DataSet
from utils.common import YOLOLoss
from utils.image import draw_boxes, draw_point, create_grid
from utils.bbox import xywh2xyxy_np
from config.config import *

RED = [0, 0, 255]
BLUE = [255, 0, 0]
GREEN = [0, 255, 0]


def visual_yolo_target(img, label, target, stride, anchor_offset):
    for i, (im, boxes) in enumerate(zip(img, label)):
        im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
        grid_img_1 = create_grid()
        grid_img_2 = create_grid()
        cv.imshow('demo', grid_img_1)
        cv.waitKey(0)
        im_label = im.copy()
        mask = np.all(boxes != [[0., 0., 0., 0.]], axis=-1)
        val_box = boxes[mask]
        if not len(val_box):
            continue
        yxc = target[target[..., 0] == i][..., 1:]
        yxc[..., :2] *= stride
        yxc[..., :2] += (stride // 2)
        im_label = draw_boxes(im_label, val_box, color=(0, 255, 0))
        for item in val_box:
            x = item[0] + (item[2]-item[0]) / 2
            y = item[1] + (item[3]-item[1]) / 2
            draw_point(grid_img_1, x, y, BLUE)
        combine_img_1 = np.concatenate([im_label, grid_img_1], axis=1)
        for j, item in enumerate(yxc):
            xy = item[..., :2][..., ::-1]
            wh = ANCHORS[int(item[..., -1]) + anchor_offset]
            xywh = np.concatenate([xy, wh], axis=-1)
            xyxy = xywh2xyxy_np(xywh)
            grid_img_2 = draw_point(grid_img_2, xy[0], xy[1], color=RED)
            im = draw_boxes(im, [xyxy], center=True)
            combine_img_2 = np.concatenate([im, grid_img_2], axis=1)
            combine_img = np.concatenate([combine_img_1, combine_img_2], axis=0)
            cv.imshow('combine_img', combine_img)
            cv.waitKey(0)


large_strid = 32
small_strid = 16
b = 1
tfrecord_path = 'data/tfrecord/train/'

train_set = DataSet(b, tfrecord_path, mode='train', params={})
f_large = np.zeros((b, 13, 13, 45), dtype=np.float32)
f_small = np.zeros((b, 26, 26, 45), dtype=np.float32)
yolo_large = YOLOLoss(ANCHORS, mask=[3, 4, 5], num_classes=10, grid_size=13)
yolo_small = YOLOLoss(ANCHORS, mask=[1, 2, 3], num_classes=10, grid_size=26)

f_large_box = f_large.reshape((b, 13, 13, 3, 15))[..., :4]
f_small_box = f_small.reshape((b, 26, 26, 3, 15))[..., :4]

xy = np.meshgrid(np.arange(13), np.arange(13))
xy = np.stack(xy, axis=-1).astype(np.float)
wh = ANCHORS[3:]
center_xy = xy * 32 + 32 / 2
center_xy = np.tile(center_xy[np.newaxis:, :, np.newaxis, :], [b, 1, 1, 3, 1])
wh = np.broadcast_to(wh, center_xy.shape)
xywh = np.concatenate([center_xy, wh], axis=-1).astype(np.float32)
xywh /= IMG_SIZE

fig, axes = plt.subplots(2, 2, constrained_layout=False)
for img, _, label in train_set():
    img = img.numpy()
    label = label.numpy()
    *_, ignore_mask, _, bnyxc = yolo_large.build_target(xywh, label)
    ignore_idx = np.argwhere(ignore_mask.numpy())
    bnyxc = bnyxc.numpy()
    byxc = bnyxc[..., [0, 2, 3, 4]]
    label = xywh2xyxy_np(label[..., :4]) * IMG_SIZE
    visual_yolo_target(img, label, byxc, 32, 3)
