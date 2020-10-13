import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

from utils.bbox import xywh2xyxy

COLORS = {
    'red': [40, 39, 214],
    'pure_red': [0, 0, 255],
    'green': [119, 158, 27],
    'pure_green': [0, 255, 0],
    'blue': [184, 126, 55],
    'pure_blue': [255, 0, 0],
    'purple':[162, 82, 83],
    'gray': [170, 170, 170]
}


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
               cls=None,
               classes=None,
               color=(40, 39, 214),
               mask=False,
               center=False,
               center_color=(0, 255, 0),
               thickness=4):
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if not isinstance(boxes, np.ndarray):
        boxes = np.asarray(boxes, dtype=np.int)
    colors = None
    if classes is not None:
        colors = {c: get_color(c) for c in cls}
    print(cls)
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
            color = color if not classes else colors[cls[idx]]
            img = cv.rectangle(img, pt1, pt2,
                               color, thickness=thickness)
        if classes is not None:
            # 增加灰色填充bar
            pt21 = (pt1[0], pt1[1] - line_thickness)
            pt22 = (pt2[0], pt1[1])
            img[pt21[1]:pt22[1], pt21[0]:pt22[0] + 2, :] = [184, 126, 55]
            category = str(classes[cls[idx]])
            img = cv.putText(img, category, org,
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


def get_color(idx):
    cmap = plt.get_cmap('tab20')
    colors = [[int(c * 255) for c in cmap(i)[:3]][::-1] for i in range(20)]
    return colors[idx]


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
    inter = 1
    w = grid_size * grid_stride
    h = w
    img = np.zeros([h, w, 3])
    img += [[128, 128, 128]]
    i, j = 0, 0
    while j <= h:
        img[j:j + inter, :, :] = line_color
        img[:, i:i + inter, :] = line_color
        j += grid_stride
        i = j
    return img.astype(np.uint8)


def combine_img(img_list, ncols, nrows, padding, color=None):
    if color is None:
        color = COLORS['gray']
    assert len(img_list) == ncols * nrows, 'img_list length unmatchs ncols and nrows'
    clone_img_list = []
    for img in img_list:
        padded_img = cv.copyMakeBorder(img,
                                       padding, 0, padding, 0,
                                       borderType=cv.BORDER_CONSTANT, value=color)
        clone_img_list.append(padded_img)
    height, width, c = img_list[0].shape
    ret_shape = [height * nrows + (nrows + 1) * padding,
                 width * ncols + (ncols + 1) * padding,
                 c]
    ret_img = np.zeros(ret_shape, np.uint8)
    ret_img += np.array(color, dtype=np.uint8)
    tmp_img = np.stack(clone_img_list)
    tmp_shape = [ret_shape[0] - padding, ret_shape[1] - padding, c]
    tmp_img = (tmp_img.reshape([nrows, ncols, height + padding, width + padding, c])
               .transpose([0, 2, 1, 3, 4])
               .reshape(tmp_shape))
    ret_img[:tmp_shape[0], :tmp_shape[1], :] = tmp_img
    return ret_img
