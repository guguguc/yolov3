import os
import time
import datetime
import logging

import tensorflow as tf
import tqdm

from collections import OrderedDict
from pathlib import Path
from terminaltables import SingleTable
from termcolor import colored

from main.dataloader.dataset import DataSet
from main.logger import timer
from main.evaluater import Evaluater
from utils.misc import *
from utils.util import copy_files, write_file


class Trainer:
    def __init__(self, net, train_params: dict, data_params: dict):
        self.net = net
        self.batchsize = train_params.get('batch', 16)
        self.epoches = train_params.get('epoch', 50)
        self.save_epoch = train_params.get('save_epoch', 10)
        # 初始化日志
        log_path = train_params.get('log_path')
        self.restore = train_params.get('restore', None)
        self.checkpoint_path = train_params.get('save_path', 'data/checkpoint')
        self.evaluater_log_path = train_params.get('eval_path', 'data/map')
        # 优化器配置
        self.optimizer_config = train_params.get('optimizer')
        self.optimizer, self.lr_init = get_optimizer(self.optimizer_config)
        # 计数器
        self.global_count = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.global_count_val = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.global_epoch = tf.Variable(0, dtype=tf.int64, trainable=False)
        self.warmup_epoch = 3
        self.warmup_steps = 0
        self.step_per_epoch = 0
        self.step_per_epoch_val = 0
        self.metric_names = []
        self.metrics = {}
        self.class_names = None
        self.logger = logging.getLogger(__name__)
        self.tf_logger = None
        self.dataset = None
        self.ckpt_manager = None
        self.evaluater = None
        self.is_frozen = False
        self.frozen_layers_exclude = ['conv_17', 'conv_20']
        self.frezen_epoch = 30
        self.first_eval = True
        self.init_dataset(data_params)
        self.init_evaluater()
        self.init_metrics()
        self.init_saver()
        self.init_logger(log_path)
        self.optimizer_schedule = tf.keras.experimental.CosineDecay(
            initial_learning_rate=self.lr_init,
            decay_steps=self.epoches * self.step_per_epoch - self.warmup_steps,
            alpha=1e-6)
        tf.config.run_functions_eagerly(True)

    def frozen(self):
        for layer in self.net.backbone.layers:
            if layer.name in self.frozen_layers_exclude:
                continue
            else:
                layer.trainable = False

    def unfrozen(self):
        self.net.backbone.trainable = True

    @timer
    def init_dataset(self, params):
        self.logger.info('start to load dataset!')
        train_path = params.get('train')
        val_path = params.get('val')
        self.logger.info(f'train dataset path is {train_path}\n'
                         f' [*] val dataset path is {val_path}')
        name_path = params.get('name')
        self.class_names = [
            line.strip() for line in open(name_path).readlines()
        ]
        train_set = DataSet(batch_size=self.batchsize,
                            tf_record_path=train_path,
                            mode='train',
                            params={})()
        val_set = DataSet(batch_size=self.batchsize,
                          tf_record_path=val_path,
                          mode='val',
                          params={})()
        self.dataset = dict(train=train_set, val=val_set)
        len_data_train = 0
        len_data_val = 0

        trainset_start_time = time.time()
        for img, img_id, label in train_set:
            len_data_train += 1
        trainset_end_time = time.time()

        valset_start_time = time.time()
        for img, img_id, label in val_set:
            len_data_val += 1
        valset_end_time = time.time()

        trainset_load_time = trainset_end_time - trainset_start_time
        valset_load_time = valset_end_time - valset_start_time
        trainset_load_per_step = 1000 * trainset_load_time / len_data_train
        valset_load_per_step = 1000 * valset_load_time / len_data_val

        self.step_per_epoch = len_data_train
        self.step_per_epoch_val = len_data_val
        self.warmup_steps = self.step_per_epoch * self.warmup_epoch
        self.logger.info(
            f'batch size is {self.batchsize}\n'
            f' [*] total steps in one train epoch is {len_data_train}\n'
            f' [*] total steps in one val epoch is {len_data_val}\n'
            f' [*] total epoch is {self.epoches}\n'
            f' [*] train set load time is {trainset_load_time:.2f} s\n'
            f' [*] train set load time per step is {trainset_load_per_step} ms\n'
            f' [*]   val set load time is {valset_load_time:.2f} s\n'
            f' [*]   val set load time per step is {valset_load_per_step} ms')

    def init_evaluater(self):
        self.evaluater = Evaluater(log_path='data/map/',
                                   class_name=self.class_names,
                                   iou_threshold=0.5,
                                   score_threshold=0.25)

    def init_metrics(self):
        self.metric_names = ['loss_box', 'loss_obj', 'loss_noobj', 'loss_cls']
        self.metrics = {'train': OrderedDict(), 'val': OrderedDict()}
        for mode in self.metrics.keys():
            for metric in self.metric_names:
                self.metrics[mode][metric] = keras.metrics.Mean(
                    name=metric, dtype=tf.float32)

    def init_saver(self):
        self.logger.info('started to init checkpointer manager')
        ckpt = tf.train.Checkpoint(optimizer=self.optimizer,
                                   net=self.net.backbone,
                                   step=self.global_count)
        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint=ckpt, directory=self.checkpoint_path, max_to_keep=10)
        self.logger.info(f'check point path is {self.checkpoint_path}')
        if self.restore:
            self.logger.info(f'restore ckpt {self.restore}')
            ckpt.restore(save_path=self.restore)
            self.logger.info(f'succeed to restore')
            self.optimizer.lr = self.lr_init
            self.global_count.assign(0)

    def init_logger(self, log_path):
        self.logger.info('start to init tensorboard logger')
        log_path = str(Path(log_path).resolve().absolute())
        t = datetime.datetime.now().strftime('%Y-%m-%d-%H_%M_%S')
        train_log_path = os.path.join(log_path, 'net', t, 'train')
        val_log_path = os.path.join(log_path, 'net', t, 'val')
        self.logger.info(f'train log path is {train_log_path}\n'
                         f'  val log path is {val_log_path}')
        train_log_writer = tf.summary.create_file_writer(logdir=train_log_path)
        val_log_writer = tf.summary.create_file_writer(logdir=val_log_path)
        self.tf_logger = dict(train=train_log_writer, val=val_log_writer)

    def update_metrics(self, losses, mode='train'):
        for metric, loss in zip(self.metrics[mode].values(), losses):
            metric.update_state(loss)

    def reset_metrics(self, mode='train'):
        for metric in self.metrics[mode].values():
            metric.reset_states()

    def show_metric(self):
        columns = ['loss type', 'train', 'val']
        data = [columns]
        for train_loss, val_loss in zip(self.metrics['train'].values(),
                                        self.metrics['val'].values()):
            data.append([
                colored(train_loss.name, 'green'),
                f'{train_loss.result():.4f}', f'{val_loss.result():.4f}'
            ])
        table = SingleTable(data, title='loss info')
        self.logger.info(f'\n{table.table}')

    def write_logger(self, step, mode):
        logger = self.tf_logger[mode]
        with logger.as_default():
            for metric in self.metrics[mode].values():
                tf.summary.scalar(name=metric.name,
                                  data=metric.result(),
                                  step=step)

    def fine_tune(self, epoch):
        if epoch < self.frezen_epoch and not self.is_frozen:
            self.logger.info(f'frozen backbone layer in epoch {epoch}')
            self.frozen()
            self.is_frozen = True
        elif epoch >= self.frezen_epoch and self.is_frozen:
            self.logger.info(f'unfrozen backbone layer in epoch {epoch}')
            self.unfrozen()
            self.is_frozen = False

    def lr_adjust(self):
        if tf.less(self.global_count, self.warmup_steps):
            lr = tf.divide(self.global_count,
                           self.warmup_steps) * self.lr_init
        else:
            lr = self.optimizer_schedule(
                tf.math.subtract(self.global_count, self.warmup_steps))
        lr = tf.cast(lr, tf.float32)
        self.optimizer.lr.assign(lr)
        self.logger.debug(f'lr is {lr:.4f}')

    @tf.function(input_signature=[tf.TensorSpec([16, 416, 416, 3], tf.float32),
                                  tf.TensorSpec([16, None, 5], tf.float32)])
    def train_step(self, img, label):
        with tf.GradientTape() as tape:
            losses = self.net.forward(img, label, training=True)
            total_loss = tf.math.reduce_sum(losses)
        grads = tape.gradient(total_loss, self.net.backbone.trainable_variables)
        # tf.print(self.net.backbone.get_layer('conv_001').variables[3])
        self.optimizer.apply_gradients(zip(grads, self.net.backbone.trainable_variables))
        self.update_metrics(losses, mode='train')
        self.write_logger(step=self.global_count, mode='train')
        self.global_count.assign_add(1)

    @tf.function(input_signature=[tf.TensorSpec([16, 416, 416, 3], tf.float32),
                                  tf.TensorSpec([16, None, 5], tf.float32)])
    def test_step(self, img, label):
        losses = self.net.forward(img, label, training=False)
        self.update_metrics(losses, mode='val')
        self.write_logger(step=self.global_count_val, mode='val')
        self.global_count_val.assign_add(1)

    @tf.function
    def eval_step(self, img):
        return self.net.predict(img)

    def train(self):
        self.logger.info(
            f'start to train model in step {self.global_count.numpy()}')
        dataset = self.dataset['train']
        best_map = 0
        for epoch in tf.range(self.epoches):
            # 迁移学习
            # self.fine_tune(epoch)
            self.reset_metrics(mode='train')
            start = time.perf_counter()
            bar = tqdm.tqdm(dataset, total=self.step_per_epoch)
            for batch_img, _, batch_label in bar:
                self.train_step(batch_img, batch_label)
                # 学习率调整
                self.lr_adjust()
            self.global_epoch.assign_add(1)
            end = time.perf_counter()
            internal = int((end - start) * 1000)
            save_path = self.ckpt_manager.save()
            self.logger.info(
                f'epoch {epoch} started!, last epoch cost {internal // 60} sec '
                f'every step cost {internal // self.step_per_epoch} ms '
            )
            self.logger.info(f'checkpoint saved in {save_path}')
            # 验证
            if tf.math.mod(epoch, 2) == 0:
                self.test()
            # 评估MAP
            mean_ap, ap = self.eval()
            if mean_ap > best_map:
                best_map = mean_ap
                best_path = os.path.join(self.checkpoint_path, 'best')
                pattern = save_path + '*'
                copy_files(pattern, best_path)
                write_file(os.path.join(best_path, 'best'), str(best_map))
                self.logger.info(f'[*] best map is {best_map}')
            # 输出loss信息
            self.show_metric()

    def test(self):
        self.logger.info('start to valid val dataset')
        val_set = self.dataset['val']
        self.reset_metrics(mode='val')
        bar = tqdm.tqdm(val_set, total=self.step_per_epoch_val)
        for img, _, label in bar:
            self.test_step(img, label)
        self.logger.info('valid val dataset finished!')

    @timer
    def eval(self):
        val_set = self.dataset['train']
        evaluator = self.evaluater
        self.logger.info('start to eval map')
        bar = tqdm.tqdm(val_set, total=self.step_per_epoch)
        t1 = time.perf_counter()
        for img, img_id, label in bar:
            label_mask = tf.reduce_all(tf.equal(label, tf.zeros([1])), axis=-1)
            label_mask = tf.logical_not(label_mask)
            valid_gt_indices = tf.math.bincount(
                tf.cast(tf.where(label_mask), tf.int32)[..., 0])
            gt_boxes, gt_cls = tf.split(label, [4, 1], axis=-1)
            nms_boxes, nms_score, nms_cls, valid_nms_indices = self.eval_step(img)
            if self.first_eval:
                evaluator.update_gt(gt_boxes, gt_cls, valid_gt_indices, img_id)
            evaluator.update_pred(nms_boxes, nms_cls, nms_score,
                                  valid_nms_indices, img_id)
        t2 = time.perf_counter()
        internal = int(t2 - t1) * 1000
        step_internal = internal // self.step_per_epoch
        self.logger.info(f'eval cost {internal} ms, '
                         f'every step cost {step_internal} ms')
        mean_ap, ap = evaluator.eval()
        self.first_eval = False
        return mean_ap, ap
