from main.evaluater import Evaluater

if __name__ == '__main__':
    gt_path = 'data/map/gt.csv'
    pred_path = 'data/map/pred.csv'
    cls_name = [line.strip() for line in open('data/data.names').readlines()]
    evaluator = Evaluater(pred_path=pred_path,
                          gt_path=gt_path,
                          class_name=cls_name)
    mean_ap, ap = evaluator.eval()
    print(mean_ap)
