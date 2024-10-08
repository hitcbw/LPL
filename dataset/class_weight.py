import os
from typing import List, Optional

import numpy as np
import pandas as pd
import torch


__all__ = ["get_pos_weight", "get_class_weight"]

from utils.class_id_map import get_n_classes

modes = ["training", "trainval"]

def get_class_nums(
    dataset: str,
    split: int = 1,
    dataset_dir: str = "./dataset/",
    csv_dir: str = "./csv",
    mode: str = "trainval",
) -> List[int]:

    assert (
        mode in modes
    ), "You have to choose 'training' or 'trainval' as the dataset mode."

    if mode == "training": # 根据指定的 mode 读取相应的 CSV 文件
        df = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv").format(split))
    elif mode == "trainval":
        df1 = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv".format(split)))
        df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
        df = pd.concat([df1, df2])

    n_classes = get_n_classes(dataset, dataset_dir) # 获取数据集中的类别总数

    nums = [0 for i in range(n_classes)] # 初始化一个长度为类别总数的列表，用于存储每个类别的样本数量
    for i in range(len(df)): # 遍历数据集中的每个样本
        label_path = dataset_dir + df.iloc[i]["label"]
        label = np.load(label_path).astype(np.int64) # 从文件中加载标签数据
        num, cnt = np.unique(label, return_counts=True) # 统计标签中每个类别的样本数量
        for n, c in zip(num, cnt):
            nums[n] += c # 将每个类别的样本数量累加到相应的位置

    return nums # 返回每个类别的样本数量列表


def get_class_weight(
    dataset: str,
    split: int = 1,
    dataset_dir: str = "./dataset",
    csv_dir: str = "./csv",
    mode: str = "trainval",
) -> torch.Tensor:
    """
    Class weight for CrossEntropy
    Class weight is calculated in the way described in:
        D. Eigen and R. Fergus, “Predicting depth, surface normals and semantic labels with a common multi-scale convolutional architecture,” in ICCV,
        openaccess: https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Eigen_Predicting_Depth_Surface_ICCV_2015_paper.pdf
    """
    # 调用 get_class_nums 函数获取数据集中各类别的样本数量
    nums = get_class_nums(dataset, split, dataset_dir, csv_dir, mode)

    class_num = torch.tensor(nums) # 将类别样本数量转换为 PyTorch 的张量
    total = class_num.sum().item() # 计算总样本数量
    frequency = class_num.float() / total # 计算每个类别的相对频率
    median = torch.median(frequency) # 计算频率的中值
    class_weight = median / frequency # 计算类别权重，即中值与频率的比值

    return class_weight # 返回计算得到的类别权重


def get_pos_weight(
    dataset: str,
    split: int = 1,
    csv_dir: str = "./csv",
    mode: str = "trainval",
    norm: Optional[float] = None,
    base_dir = ''
) -> torch.Tensor:
    """
    pos_weight for binary cross entropy with logits loss
    pos_weight is defined as reciprocal of ratio of positive samples in the dataset
    """

    assert (
        mode in modes
    ), "You have to choose 'training' or 'trainval' as the dataset mode"

    if mode == "training":
        df = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv").format(split))
    elif mode == "trainval":
        df1 = pd.read_csv(os.path.join(csv_dir, dataset, "train{}.csv".format(split)))
        df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
        df = pd.concat([df1, df2])

    n_classes = 2  # boundary or not 二分类
    nums = [0 for i in range(n_classes)] # 初始化一个长度为类别总数的列表，用于存储每个类别的样本数量
    for i in range(len(df)):
        label_path = base_dir + df.iloc[i]["boundary"]
        label = np.load(label_path, allow_pickle=True).astype(np.int64)
        num, cnt = np.unique(label, return_counts=True) # 统计标签中每个类别的样本数量
        for n, c in zip(num, cnt):
            nums[n] += c # 将每个类别的样本数量累加到相应的位置

    pos_ratio = nums[1] / sum(nums)
    pos_weight = 1 / pos_ratio

    if norm is not None:
        pos_weight /= norm

    return torch.tensor(pos_weight) #权重=总数/边界数
