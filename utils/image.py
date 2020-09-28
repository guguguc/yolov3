import random

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt

from utils.bbox import xywh2xyxy


def draw_img(img, nrows=1, ncols=1, ticks=False, figsize=(5, 10)):
    if np.ndim(img) == 3:
        img = np.expand_dims(img, axis=0)
    if img.shape[0] != nrows * ncols:
        raise ValueError(f'img shape {img.shape} unmatched nrows {nrows}, ncols {ncols}')

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, tight_layout=True, figsize=figsize)

    for im, ax in zip(img, np.ravel(axes)):
        ax.imshow(im)
    if not ticks:
        for ax in np.ravel(axes):
            ax.set_xticks([])
            ax.set_yticks([])
    return fig, axes


def draw_boxes(img,
               boxes,
               classes=None,
               color=(255, 0, 0),
               mask=False,
               center=False,
               center_color=(0, 255, 0),
               thickness=2):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes, dtype=np.int)
    font_color = (0, 0, 255)
    # line_color = (128, 128, 128)
    line_thickness = 12
    font = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    line = cv.LINE_4
    for idx, box in enumerate(boxes):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        org = (pt1[0], pt1[1] - 4)
        if mask:
            # alpaha为src1透明度，beta为src2透明度
            alpha, beta, gamma = 1, 0.1, 0
            mask_img = np.zeros(img.shape, dtype=img.dtype)
            # color = gen_random_color()
            # print(color)
            mask_img = cv.rectangle(mask_img, pt1, pt2,
                                    color=color, thickness=-1)
            img = cv.addWeighted(img, alpha, mask_img, beta, gamma)
        else:
            img = cv.rectangle(img, pt1, pt2,
                               color, thickness=thickness)
        if classes is not None:
            # 增加灰色填充bar
            pt21 = (pt1[0], pt1[1] - line_thickness)
            pt22 = (pt2[0], pt1[1])
            img[pt21[1]:pt22[1], pt21[0]:pt22[0] + 2] = 128
            cls = str(classes[idx])
            img = cv.putText(img, cls, org,
                             font, font_scale, font_color,
                             1, line, False)
        if center:
            pt_center = (int((pt1[0] + pt2[0]) / 2), int((pt1[1] + pt2[1]) / 2))
            img = draw_point(img, pt_center[0], pt_center[1],
                             color=center_color, thickness=7)

    return img


def draw_grid(img, row, col):
    img_h, img_w = img.shape[:2]
    x = (np.arange(row) * (img_w / col)).astype(np.int)
    y = (np.arange(col) * (img_h / row)).astype(np.int)
    color = (128, 128, 128)
    thickness = 1
    for i in y:
        pt1 = (0, int(i))
        pt2 = (img_w, int(i))
        img = cv.line(img, pt1, pt2, color, thickness=thickness)
    for j in x:
        pt1 = (int(j), 0)
        pt2 = (int(j), img_w)
        img = cv.line(img, pt1, pt2, color, thickness=thickness)
    return img


def draw_point(img, x, y, color=(0, 1, 0), thickness=4):
    pt = (int(x), int(y))
    img = cv.circle(img, pt, 1, color=color, thickness=thickness)
    return img


def gen_random_color():
    c1 = random.randint(a=0, b=100)
    c2 = random.randint(a=0, b=100)
    c3 = random.randint(a=0, b=100)
    return c1, c2, c3


def visiual_ground_truth(img, obj_mask, anchors, anchor_mask, stride, scale_num=3):
    obj_indices = np.argwhere(obj_mask)
    img = np.tile(img, [scale_num, 1, 1, 1])
    # 标记正样本
    for byxc in obj_indices:
        y, x, c = byxc
        scale = int(c)
        wh = anchors[anchor_mask[scale]]
        box = [[x * stride, y * stride, wh[0], wh[1]]]
        box = xywh2xyxy(box)
        img[scale] = draw_boxes(img[scale], box,
                                center=True, color=(0, 35, 0),
                                center_color=(255, 0, 0),
                                thickness=4)
    return img


def create_grid(grid_size=13, grid_stride=32):
    line_color = [0, 0, 0]
    inter = 2
    w = grid_size * grid_stride
    h = w
    img = np.zeros([h + inter, w + inter, 3])
    img += [[128, 128, 128]]
    i, j = 0, 0
    #while j <= h:
    #    print(j, j+inter)
    #    img[j:j + inter, :, :] = line_color
    #    print(img[j:j+1, :, :].shape)
    #    img[:, i:i + inter, :] = line_color
    #    j += grid_stride
    #    i = j
    return img


if __name__ == '__main__':
    im = create_grid()
    cv.imshow('demo', im)
    cv.waitKey(0)
