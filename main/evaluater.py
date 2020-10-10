import logging
import os
import numpy as np
import pandas as pd

from utils.bbox import calc_area_pd, xywh2xyxy
from utils.util import calc_ap
from main.logger import timer


class Evaluater:
    def __init__(self,
                 class_name: list,
                 iou_threshold=0.5,
                 score_threshold=0.25,
                 max_det=300,
                 gt_path=None,
                 pred_path=None,
                 log_path=None):
        """
        :param class_name: 类别名称列表
        :param iou_threshold: 预测边界框与标签边界框的IOU小于此阈值则认为是FP
        :param score_threshold: 类别概率小于此阈值的样本不进行TP的统计
        :param gt_path: 标签csv文件的路径
        :param pred_path: 预测csv文件的路径
        :param log_path: 如果没有提供gt_path, pred_path,
               则认为工作在训练模式，其gt, pred数据将被dump到此目录
        """
        super(Evaluater, self).__init__()
        self.iou_threshold = iou_threshold
        self.score_threshold = score_threshold
        self.max_det = max_det
        self.class_num = len(class_name)
        self.class_name = class_name
        # 标识evluator的工作模式
        self.mode = None
        # gt与pred的Dataframe字段
        gt_columns = ['img_id', 'xmin', 'ymin', 'xmax', 'ymax', 'cls']
        pred_columns = gt_columns + ['score']
        # 评估结果的Dataframe字段
        metric_colums = ['TP', 'FP', 'TP+FP', 'TP+FN',
                         'Precision', 'Recall',
                         'Ap', 'F1 score']
        self.box_filed = ['xmin', 'ymin', 'xmax', 'ymax']
        self.metric_matrix = pd.DataFrame(columns=metric_colums,
                                          index=pd.Index(class_name,
                                                         name='category'))
        self.gt = None
        self.pred = None
        self.logger = logging.getLogger(__name__)

        # 初始化gt, pred
        if log_path:
            self.mode = 'online'
            self.log_path = log_path
            self.log_gt_path = os.path.join(self.log_path, 'gt.csv')
            self.log_pred_path = os.path.join(self.log_path, 'pred.csv')
            self.gt = pd.DataFrame(columns=gt_columns)
            self.pred = pd.DataFrame(columns=pred_columns)
        elif pred_path and gt_path:
            self.mode = 'offline'
            self.pred_path = pred_path
            self.gt_path = gt_path
            self.gt = pd.read_csv(gt_path)
            self.pred = pd.read_csv(pred_path)
            assert (self.gt.columns == gt_columns).all(), \
                f"error columns {self.gt.columns}, it should be {gt_columns}"
            assert (self.pred.columns == pred_columns).all(), \
                f"error columns {self.pred.columns}, it should be {pred_columns}"
        else:
            raise ValueError('[gt_path, pred_path] and [log_path] must be offered either')

    def update_gt(self, boxes, classes, valid_indices, img_id):
        """将gt boxes, gt classes写入文本文件，批量写入
        :param boxes: Tensor of shape (batchsize, M, 4)
        :param classes: Tensor of shape (batchsize, M, 1)
        :param valid_indices: Tensor of shape (batchsize,)， 有效索引
        :param img_id: Tensor of shape (batchsize,) Type:tf.string
        """
        batch_size = boxes.shape[0]
        boxes = xywh2xyxy(boxes).numpy()
        classes = classes.numpy().astype(np.int32)
        img_id = img_id.numpy()
        valid_indices = valid_indices.numpy()
        gt_info = dict(img_id=[], cls=[], xmin=[], ymin=[], xmax=[], ymax=[])
        for idx, indices in zip(range(batch_size), valid_indices):
            box = boxes[idx, :indices]
            cls = list(classes[idx, :indices].ravel())
            im_id = [str(img_id[idx], encoding='utf8')] * indices
            gt_info['img_id'].extend(im_id)
            gt_info['cls'].extend(cls)
            for i, field in enumerate(self.box_filed):
                gt_info[field].extend(list(box[:, i]))
        self.gt = self.gt.append(pd.DataFrame(gt_info), ignore_index=True)

    def update_pred(self, boxes, classes, scores, valid_indices, img_id):
        batch_size = boxes.shape[0]
        boxes = boxes.numpy()
        classes = classes.numpy().astype(np.int32)
        img_id = img_id.numpy()
        scores = scores.numpy()
        valid_indices = valid_indices.numpy()
        pred_info = dict(img_id=[], cls=[], xmin=[], ymin=[], xmax=[], ymax=[], score=[])
        for idx, indices in zip(range(batch_size), valid_indices):
            box = boxes[idx, :indices]
            cls = list(classes[idx, :indices].ravel())
            score = list(scores[idx, :indices].ravel())
            im_id = [str(img_id[idx], encoding='utf8')] * indices
            pred_info['img_id'].extend(im_id)
            pred_info['cls'].extend(cls)
            pred_info['score'].extend(score)
            for i, field in enumerate(self.box_filed):
                pred_info[field].extend(list(box[:, i]))
        self.pred = self.pred.append(pd.DataFrame(pred_info), ignore_index=True)

    def _prepare(self):
        for field in self.box_filed:
            self.gt[field] = self.gt[field].astype(np.float32)
            self.pred[field] = self.pred[field].astype(np.float32)
        self.gt['cls'] = self.gt['cls'].astype('int')
        self.pred['cls'] = self.pred['cls'].astype('int')
        self.logger.info('ground truth num is %d, predication num is %d',
                         len(self.gt.index), len(self.pred.index))
        # 将pred中相同img_id内部的分数从大到小排序
        self.pred = self.pred.sort_values(by=['img_id', 'score'],
                                          ascending=[True, False],
                                          ignore_index=True)
        self.pred = self.pred.groupby('img_id').head(self.max_det).reset_index(drop=True)
        self.gt = self.gt.sort_values(by=['img_id'], ignore_index=True)
        self.gt['id'] = list(self.gt.index)
        self.pred['id'] = list(self.pred.index)

    def _set_metric_matrix(self, info: pd.DataFrame, label: pd.DataFrame):
        statistic = info.groupby('cls').sum()
        statistic = statistic.reindex([i for i in range(self.class_num)], fill_value=0)
        field = ['TP', 'FP', 'FP[iou unmatch]', 'FP[cls unmatch]']
        self.metric_matrix[field] = statistic.to_numpy()
        self.metric_matrix['TP+FN'] = np.bincount(label['cls'])
        self.metric_matrix['TP+FP'] = self.metric_matrix['TP'] + self.metric_matrix['FP']
        self.metric_matrix['Precision'] = self.metric_matrix['TP'] / self.metric_matrix['TP+FP']
        self.metric_matrix['Recall'] = self.metric_matrix['TP'] / self.metric_matrix['TP+FN']
        precision = self.metric_matrix['Precision']
        recall = self.metric_matrix['Recall']
        self.metric_matrix['F1 score'] = (2 * precision * recall) / (precision + recall + 1e-6)

    def _reset_data(self):
        self.pred = self.pred.iloc[0:0]
        self.metric_matrix.iloc[:, :] = np.nan

    @timer
    def eval(self):
        self._prepare()
        gt = self.gt
        pred = self.pred
        # 对于相同的img_id与cls pred有N个， gt有M个 -> N X M
        combined_info = pd.merge(pred, gt, on=['img_id', 'cls'], suffixes=['_pred', '_gt'])
        combined_info['xmin_inter'] = combined_info[['xmin_pred', 'xmin_gt']].max(1)
        combined_info['ymin_inter'] = combined_info[['ymin_pred', 'ymin_gt']].max(1)
        combined_info['xmax_inter'] = combined_info[['xmax_pred', 'xmax_gt']].min(1)
        combined_info['ymax_inter'] = combined_info[['ymax_pred', 'ymax_gt']].min(1)
        inter_area = calc_area_pd(combined_info, ['xmin_inter', 'ymin_inter',
                                                  'xmax_inter', 'ymax_inter'])
        pred_area = calc_area_pd(combined_info, ['xmin_pred', 'ymin_pred',
                                                 'xmax_pred', 'ymax_pred'])
        gt_area = calc_area_pd(combined_info, ['xmin_gt', 'ymin_gt',
                                               'xmax_gt', 'ymax_gt'])
        combined_info['iou'] = inter_area / (pred_area + gt_area - inter_area)
        assert (combined_info['iou'] >= 0).all()
        assert (combined_info['iou'] <= 1).all()
        combined_info = combined_info.sort_values(by='iou').groupby('id_pred', as_index=False).last()
        unique_mask = ~combined_info.duplicated(subset='id_gt', keep='first')
        # 类别预测正确并且最大IOU大于阈值
        match_mask = combined_info['iou'] > self.iou_threshold
        # 类别预测正确但是最大IOU小于阈值
        iou_unmatch_fp_mask = unique_mask & ~match_mask
        tp_mask = unique_mask & match_mask
        # combined_info['tp'] = unique_mask & match_mask
        # 超过类别概率超过分数阈值的pred
        pred[['tp', 'fp', 'fp[iou]', 'fp[cls]']] = [False, True, False, False]
        tp_index = combined_info['id_pred'][tp_mask]
        iou_unmatch_fp_index = combined_info['id_pred'][iou_unmatch_fp_mask]
        tp_mask = pred['id'].isin(tp_index)
        iou_unmatch_fp_mask = pred['id'].isin(iou_unmatch_fp_index)
        cls_unmatch_fp_mask = ~tp_mask & ~iou_unmatch_fp_mask
        pred.loc[tp_mask, 'tp'] = True
        pred.loc[tp_mask, 'fp'] = False
        pred.loc[iou_unmatch_fp_mask, 'fp[iou]'] = True
        pred.loc[cls_unmatch_fp_mask, 'fp[cls]'] = True
        thresh_pred = pred.query(f'score>{self.score_threshold}')[['tp', 'fp',
                                                                   'fp[iou]',
                                                                   'fp[cls]',
                                                                   'cls']]
        # 填充可能预测缺失的类别
        self.logger.debug(f'conf > {self.score_threshold} num is {len(thresh_pred)}')
        self._set_metric_matrix(thresh_pred, gt)
        pred = pred.sort_values(by=['score'], ascending=[False], ignore_index=True)
        y_truth = pred['cls'].copy()
        y_truth[pred['fp']] = -1
        ap = calc_ap(pred['cls'], y_truth, self.metric_matrix['TP+FN'], self.class_num)
        mean_ap = np.mean(ap)
        self.metric_matrix['Ap'] = ap
        self.logger.info(f'\n{self.metric_matrix.to_string()}')
        self.logger.info(f'\nmean ap is {mean_ap:.4f}')
        self._reset_data()
        ap = {name: accu for name, accu in zip(self.class_name, ap)}
        return mean_ap, ap

    def print(self):
        print(self.pred)
        print(self.gt)
