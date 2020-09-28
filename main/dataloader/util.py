import numpy as np
from dataloader.tf_example_encoder import TfExampleEncoder


def create_tfrecord(imgs_path: str, labels: list, classes_name: list, out_path, max_num):
    """
    :param imgs_path: 图像集合的路径列表 list [str]
    :param labels: 图像标签 list [[box1, box2, ...], ...], 顶级维度与imgs_path一致
    :param classes_name: 类别名称列表
    :param out_path: 输出路径， str or list
    :param max_num: 单个tfrecord文件最大example数目
    :return: None
    """
    if isinstance(out_path, str):
        out_path = [out_path]
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels, dtype=object)
    if not isinstance(imgs_path, np.ndarray):
        imgs_path = np.array(imgs_path)
    # 打乱数据集
    imgs_num = len(imgs_path)
    print(f'total img num is {imgs_num}')
    indice = np.arange(imgs_num)
    np.random.shuffle(indice)
    imgs = imgs_path[indice]
    labels = labels[indice]
    class_id = labels[:, 4]
    idx = 0
    for path in out_path:
        count = 0
        with tf.io.TFRecordWriter(path) as wp:
            while idx < imgs_num and count < max_num:
                boxes = np.array(labels[idx])[:, :4]
                obj_id = np.array(labels[idx])[:, 4]
                obj_name = [classes_name[id_ - 1] for id_ in obj_id]
                serialized_example = TfExampleEncoder.encode(imgs[idx], boxes, obj_id, obj_name)
                wp.write(serialized_example)
                idx += 1
                count += 1
        print(f'succeed to write to {path}, total {count} data')
