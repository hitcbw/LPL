import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision.transforms.transforms import Compose

__all__ = ["ActionSegmentationDataset", "collate_fn"]

dataset_names = ["PKU-subject", "PKU-view", "LARA", "TCG"]
modes = ["training", "validation", "trainval", "test"]


def get_displacements(sample):
    # input: C, T, V, M
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))

    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(
        axis=0) > 0  # 计算每个时间步的有效帧数，即不全为零的帧数 维度为时间帧数（其中为true的即有效，false的表示此帧全是0）
    start = validFrames.argmax()  # 返回最大索引（第一个为真的索引）
    end = len(validFrames) - validFrames[::-1].argmax()  # 拿到最后一个为真的索引
    sample = sample[:, start:end, :, :]  # 拿到截取有效部分后的特征

    t = sample.shape[1]
    # Shape: C, t-1, V, M
    disps = sample[:, 1:, :, :] - sample[:, :-1, :, :]  # 计算位移数组 disps，通过从当前时间步的样本中减去前一个时间步的样本
    # Shape: C, T, V, M
    final_sample[:, start:end - 1, :, :] = disps  # 还是保证和原t一致，因此最后一维时间维度补零

    return final_sample


def get_relative_coordinates(sample,
                             references=(0)):
    # input: C, T, V, M
    # references=(4, 8, 12, 16)
    C, T, V, M = sample.shape
    final_sample = np.zeros((C, T, V, M))

    validFrames = (sample != 0).sum(axis=3).sum(axis=2).sum(
        axis=0) > 0  # 计算每个时间步的有效帧数，即不全为零的帧数 维度为时间帧数（其中为true的即有效，false的表示此帧全是0）
    start = validFrames.argmax()
    end = len(validFrames) - validFrames[::-1].argmax()
    sample = sample[:, start:end, :, :]  # 拿到截取有效部分后的特征

    C, t, V, M = sample.shape
    rel_coords = []
    # for i in range(len(references)):
    ref_loc = sample[:, :, references, :]  # 创建一个空列表 rel_coords，用于存储每个参考节点的相对坐标结果
    coords_diff = (sample.transpose((2, 0, 1, 3)) - ref_loc).transpose((1, 2, 0, 3))  # 所有节点的特征都减去这个节点的特征
    rel_coords.append(coords_diff)

    # Shape: C, t, V, M
    rel_coords = np.vstack(rel_coords)
    # Shape: C, T, V, M
    final_sample[:, start:end, :, :] = rel_coords
    return final_sample


class ActionSegmentationDataset(Dataset):  # 数据集
    """ Action Segmentation Dataset """

    def __init__(
            self,
            dataset: str,
            transform: Optional[Compose] = None,
            mode: str = "training",
            split: int = 1,
            dataset_dir: str = "./dataset",
            csv_dir: str = "./csv",
    ) -> None:
        super().__init__()
        """
            Args:
                dataset: the name of dataset
                transform: torchvision.transforms.Compose([...])
                mode: training, validation, test
                split: which split of train, val and test do you want to use in csv files.(default:1)
                csv_dir: the path to the directory where the csv files are saved
        """

        assert (
                dataset in dataset_names
        ), "You have to choose dataset."

        if mode == "training":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            )  # 拿到是（NUM，3）文件，其中三个分别是特征，标签和边界标签
        elif mode == "validation":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "val{}.csv".format(split))
            )
        elif mode == "trainval":
            df1 = pd.read_csv(
                os.path.join(csv_dir, dataset, "train{}.csv".format(split))
            )
            df2 = pd.read_csv(os.path.join(csv_dir, dataset, "val{}.csv".format(split)))
            self.df = pd.concat([df1, df2])
        elif mode == "test":
            self.df = pd.read_csv(
                os.path.join(csv_dir, dataset, "test{}.csv".format(split))
            )
        else:
            assert (
                    mode in modes
            ), "You have to choose 'training', 'trainval', 'validation' or 'test' as the dataset mode."
        #    <libs.transformer.ToTensor object at 0x7f3ed3ff3550>和<libs.transformer.TempDownSamp object at 0x7f3ed402abb0>
        self.transform = transform  # 变成tensor和采样
        self.dataset = dataset  # 数据集名
        self.dataset_dir = dataset_dir

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        feature_path = os.path.join(self.dataset_dir, self.df.iloc[idx]["feature"])
        label_path = os.path.join(self.dataset_dir, self.df.iloc[idx]["label"])
        boundary_path = os.path.join(self.dataset_dir, self.df.iloc[idx]["boundary"])  # 三者的地址

        feature = np.load(feature_path, allow_pickle=True).astype(np.float32)  # 特征（C，24000,19,M）

        if (self.dataset == 'MCFS-22') or (self.dataset == 'MCFS-130'):
            feature = feature[:, :, :2]  # t,v,c
            feature[:, :, 0] = feature[:, :, 0] / 1280 - 0.5
            feature[:, :, 1] = feature[:, :, 1] / 720 - 0.5
            feature = feature - feature[:, 8:9, :]
            feature = feature.transpose(2, 1, 0)  # t,v,c--->c,v,t

        elif (self.dataset == 'PKU-subject') or (self.dataset == 'PKU-view'):
            feature = feature.reshape(-1, 2, 25, 3).transpose(3, 0, 2, 1)  # t,m,v,c--->c,t,v,m
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature)
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3, 0, 2, 1).reshape(12, 25, -1)  # c,t,v,m--->mc,v,t

        elif (self.dataset == 'LARA'):
            disps = get_displacements(feature)  # 拿到每个时间步的位移 特征维数相同（像是微分特征）
            rel_coords = get_relative_coordinates(feature)  # 求了一个相对与一个参考节点的相对坐标特征
            feature = np.concatenate([disps, rel_coords], axis=0)  # 拼成12通道
            feature = feature.transpose(3, 0, 2, 1).reshape(12, 19, -1)  # c,t,v,m--->mc,v,t

        elif (self.dataset == 'TCG'):
            feature = feature.transpose(2, 0, 1)[...,np.newaxis] # t,m,v,c--->c,t,v,m
            disps = get_displacements(feature)
            rel_coords = get_relative_coordinates(feature)
            feature = np.concatenate([disps, rel_coords], axis=0)
            feature = feature.transpose(3, 0, 2, 1).reshape(6, 17, -1)

        label = np.load(label_path).astype(np.int64)
        boundary = np.load(boundary_path).astype(np.float32)

        if self.transform is not None:
            feature, label, boundary = self.transform([feature, label, boundary])

        sample = {
            "feature": feature,
            "label": label,
            "feature_path": feature_path,
            "boundary": boundary,
            "name": self.df.iloc[idx]["feature"].split('/')[-1].split('.')[0]
        }

        return sample


def collate_fn(sample: List[Dict[str, Any]]) -> Dict[str, Any]:
    max_length = max([s["feature"].shape[2] for s in sample])  # 拿到最长的长度

    feat_list = []
    label_list = []
    path_list = []
    boundary_list = []
    length_list = []
    name_list = []
    for s in sample:
        feature = s["feature"]
        label = s["label"]
        boundary = s["boundary"]
        feature_path = s["feature_path"]
        name_list.append(s["name"])
        _, _, t = feature.shape
        pad_t = max_length - t
        length_list.append(t)

        if pad_t > 0:
            feature = F.pad(
                feature, (0, pad_t), mode='constant', value=0.)  # 在后面填充0
            label = F.pad(label, (0, pad_t), mode='constant', value=255)  # 在后面填充255
            boundary = F.pad(boundary, (0, pad_t), mode='constant', value=0.)  # 在后面填充0

        # reshape boundary (T) => (1, T)
        boundary = boundary.unsqueeze(0)

        feat_list.append(feature)
        label_list.append(label)
        path_list.append(feature_path)
        boundary_list.append(boundary)

    # merge features from tuple of 2D tensor to 3D tensor
    features = torch.stack(feat_list, dim=0)  # （N，C，V，T）
    # merge labels from tuple of 1D tensor to 2D tensor
    labels = torch.stack(label_list, dim=0)  # （N，T）

    # merge labels from tuple of 2D tensor to 3D tensor
    # shape (N, 1, T)
    boundaries = torch.stack(boundary_list, dim=0)  # (N, 1, T)

    # generate masks which shows valid length for each video (N, 1, T)
    masks = [
        [[1 if i < length else 0 for i in range(max_length)]] for length in length_list
    ]
    masks = torch.tensor(masks, dtype=torch.bool)

    return {
        "feature": features,
        "label": labels,
        "boundary": boundaries,
        "feature_path": path_list,
        "mask": masks,
        "name":name_list
    }
