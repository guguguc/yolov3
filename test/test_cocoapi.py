import json
from main.cocoapi.pythonApi.pycocotools.coco import COCO
from main.cocoapi.pythonApi.pycocotools.cocoeval import COCOeval

gt_file = 'test/gt.json'
pred_file = 'test/pred.json'
test_file = 'data/annotations/instances_val2017.json'
info = json.load(open(test_file))
keys = ['images', 'annotations', 'categories']
print(info[keys[1]][0])
coco_gt = COCO(gt_file)
coco_dt = coco_gt.loadRes(pred_file)
evaluator = COCOeval(coco_gt, coco_dt, iouType='bbox')
evaluator.evaluate()
evaluator.accumulate()
evaluator.summarize()
