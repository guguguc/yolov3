from config.config import *
from main.dataloader.dataset import DataSet
from utils.bbox import xywh2xyxy_np
from utils.layer import YOLOLoss
from utils.visual import *


def visual_yolo_target(img, label,
                       target, stride, anchor_offset):
    grid_size = IMG_SIZE // stride
    for i, (im, boxes) in enumerate(zip(img, label)):
        im = cv.cvtColor(im, cv.COLOR_RGB2BGR)
        grid_img_1 = create_grid(grid_size=grid_size,
                                 grid_stride=stride)
        grid_img_2 = create_grid(grid_size=grid_size,
                                 grid_stride=stride)
        im_label = im.copy()
        mask = np.any(boxes != [[0., 0., 0., 0.]], axis=-1)
        val_box = boxes[mask]
        if not len(val_box):
            continue
        yxc = target[target[..., 0] == i][..., 1:]
        ignore_yxc = ignore_idx[ignore_idx[..., 0] == i][..., 1:]
        yxc[..., :2] = yxc[..., :2] * strid + strid / 2
        ignore_yxc[..., :2] = ignore_yxc[..., :2] * strid + strid / 2
        im_label = draw_boxes(im_label, val_box, color=COLORS['red'])
        # draw raw gt center point
        for item in val_box:
            x = item[0] + (item[2] - item[0]) / 2
            y = item[1] + (item[3] - item[1]) / 2
            draw_point(grid_img_1, x, y, COLORS['pure_red'])
        # draw ignore point
        for item in ignore_yxc:
            draw_point(grid_img_2, item[1], item[0], COLORS['pure_green'])
        # draw yolo gt center point and boxes
        for j, item in enumerate(yxc):
            xy = item[..., :2][..., ::-1]
            wh = ANCHORS[int(item[..., -1]) + anchor_offset]
            xywh = np.concatenate([xy, wh], axis=-1)
            xyxy = xywh2xyxy_np(xywh)
            grid_img_2 = draw_point(grid_img_2, xy[0], xy[1], color=COLORS['pure_blue'])
            im = draw_boxes(im, [xyxy], color=COLORS['blue'])
            nimg = combine_img(img_list=[im_label, im, grid_img_1, grid_img_2],
                               ncols=2, nrows=2, padding=8)
            cv.imshow(f'yolo-{grid_size}', nimg)
            cv.waitKey(0)


b = 1
strid = 32
grid_size = IMG_SIZE // strid
tfrecord_path = 'data/tfrecord/train/'

train_set = DataSet(b, tfrecord_path, mode='train', params={})
f_large = np.zeros((b, 13, 13, 45), dtype=np.float32)
f_small = np.zeros((b, 26, 26, 45), dtype=np.float32)
yolo_large = YOLOLoss(ANCHORS, mask=[3, 4, 5], num_classes=10, grid_size=13)
yolo_small = YOLOLoss(ANCHORS, mask=[1, 2, 3], num_classes=10, grid_size=26)

f_large_box = f_large.reshape((b, 13, 13, 3, 15))[..., :4]
f_small_box = f_small.reshape((b, 26, 26, 3, 15))[..., :4]

xy = np.meshgrid(np.arange(grid_size), np.arange(grid_size))
xy = np.stack(xy, axis=-1).astype(np.float)
wh = ANCHORS[3:]
center_xy = xy * strid + strid / 2
center_xy = np.tile(center_xy[np.newaxis:, :, np.newaxis, :], [b, 1, 1, 3, 1])
wh = np.broadcast_to(wh, center_xy.shape)
xywh = np.concatenate([center_xy, wh], axis=-1).astype(np.float32)
xywh /= IMG_SIZE

for img, _, label in train_set():
    img = img.numpy()
    img = (img * 255).astype(np.uint8)
    label = label.numpy()
    *_, ignore_mask, _, bnyxc = yolo_large.build_target(xywh, label)
    ignore_idx = np.argwhere(ignore_mask.numpy())
    bnyxc = bnyxc.numpy()
    byxc = bnyxc[..., [0, 2, 3, 4]]
    label = xywh2xyxy_np(label[..., :4]) * IMG_SIZE
    visual_yolo_target(img, label,
                       byxc, strid, 3)
