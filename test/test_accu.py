import glob
import json
import pathlib

import cv2 as cv
import numpy as np
import pandas as pd
from tqdm import tqdm

from utils.bbox import xywh2xyxy

img_root = str(pathlib.Path('data/dataset/val/').absolute().resolve())
class_name = [line.strip() for line in open('data/data.names').readlines()]
json_file = 'test/pred.json'


def generate_gt_coco():
    gt_info = dict()
    gt_data = generate_gt(save=False)
    keys = ['annotations', 'categories', 'images']

    gt_img_id = gt_data['img_id'].tolist()
    gt_box = gt_data[['xmin', 'ymin', 'xmax', 'ymax']].values.tolist()
    gt_box = [[xmin, ymin, xmax - xmin, ymax - ymin] for (xmin, ymin, xmax, ymax) in gt_box]
    gt_cls = gt_data['cls'].tolist()

    gt_info[keys[0]] = [
        {'image_id': int(img_id),
         'id': idx,
         'iscrowd': 0,
         'category_id': cls + 1,
         'bbox': box,
         'area': box[2] * box[3],
         } for idx, (img_id, box, cls) in enumerate(zip(gt_img_id, gt_box, gt_cls))]
    gt_info[keys[1]] = [{'id': idx + 1, 'name': name} for idx, name in enumerate(class_name)]
    gt_info[keys[2]] = [{'file_name': img_id + '.jpg',
                         'id': int(img_id)} for img_id in gt_img_id]

    json.dump(gt_info, open('gt.json', 'w+'), indent=4)


def generate_gt(save=True) -> pd.DataFrame:
    label_file = glob.glob(img_root + '\\*.txt')
    img_file = glob.glob(img_root + '\\*.jpg')
    print(img_file[0], label_file[0])
    columns = ['img_id', 'xmin', 'ymin', 'xmax', 'ymax', 'cls']
    info = {col: [] for col in columns}
    bar = tqdm(zip(label_file, img_file), total=len(img_file))
    for label_path, img_path in bar:
        img_id = label_path.split('\\')[-1][:-4]
        img_height, img_width = cv.imread(img_path).shape[:2]
        label = np.genfromtxt(label_path).reshape(-1, 5)
        cls = list(label[..., 0].astype(np.int).ravel())
        box = xywh2xyxy(label[..., 1:]).numpy()
        box[..., [0, 2]] *= img_width
        box[..., [1, 3]] *= img_height
        box = box.astype(np.int)
        xmin = list(box[..., 0].ravel())
        ymin = list(box[..., 1].ravel())
        xmax = list(box[..., 2].ravel())
        ymax = list(box[..., 3].ravel())
        info['img_id'].extend([img_id] * len(cls))
        info['cls'].extend(cls)
        info['xmin'].extend(xmin)
        info['ymin'].extend(ymin)
        info['xmax'].extend(xmax)
        info['ymax'].extend(ymax)

    gt = pd.DataFrame(info)
    if save:
        gt.to_csv('gt.csv', index=False)
    return gt


def generate_pred(file: str):
    columns = ['img_id', 'xmin', 'ymin', 'xmax', 'ymax', 'cls', 'score']
    ans = {col: [] for col in columns}
    info = json.load(open(file))
    bar = tqdm(info)
    count = 0
    for item in bar:
        img_id = str(item['image_id']).rjust(12, '0')
        xmin = round(item['bbox'][0])
        ymin = round(item['bbox'][1])
        xmax = xmin + round(item['bbox'][2])
        ymax = ymin + round(item['bbox'][3])
        cls = item['category_id'] - 1
        score = item['score']
        if score > 0.005:
            ans['img_id'].append(img_id)
            ans['xmin'].append(xmin)
            ans['ymin'].append(ymin)
            ans['xmax'].append(xmax)
            ans['ymax'].append(ymax)
            ans['score'].append(score)
            ans['cls'].append(cls)
            count += 1
    print(count)
    df = pd.DataFrame(ans)
    df.to_csv('pred.csv', index=False)


generate_pred(json_file)
generate_gt()
# generate_gt_coco()
