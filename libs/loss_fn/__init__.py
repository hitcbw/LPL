import sys
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .focal import FocalLoss
from .tmse import TMSE, GaussianSimilarityTMSE

__all__ = ["ActionSegmentationLoss", "BoundaryRegressionLoss"]


class ActionSegmentationLoss(nn.Module):
    """
    Loss Function for Action Segmentation
    You can choose the below loss functions and combine them.
        - Cross Entropy Loss (CE)
        - Focal Loss
        - Temporal MSE (TMSE)
        - Gaussian Similarity TMSE (GSTMSE)
    """
    # 创建一个损失函数的管理器类，该类用于根据用户的选择添加相应的损失函数和权重
    def __init__(
        self,
        ce: bool = True,
        focal: bool = True,
        tmse: bool = False,
        gstmse: bool = False,
        weight: Optional[float] = None,
        threshold: float = 4,
        ignore_index: int = 255,
        ce_weight: float = 1.0,
        focal_weight: float = 1.0,
        tmse_weight: float = 0.15,
        gstmse_weight: float = 0.15,
    ) -> None:
        super().__init__()
        self.criterions = []
        self.weights = []

        if ce: #交叉熵损失
            self.criterions.append(
                nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index)
            )
            self.weights.append(ce_weight)

        if focal: #Focal Loss（一种改进的交叉熵损失）
            self.criterions.append(FocalLoss(ignore_index=ignore_index))
            self.weights.append(focal_weight)

        if tmse: #TMSE（Topographic Map Similarity Enhancement）损失
            self.criterions.append(TMSE(threshold=threshold, ignore_index=ignore_index))
            self.weights.append(tmse_weight)

        if gstmse: # 添加 Gaussian Similarity TMSE 损失函数到损失函数列表中，threshold 参数用于设置相似性计算的阈值，ignore_index 参数指定忽略的类别
            self.criterions.append(
                GaussianSimilarityTMSE(threshold=threshold, ignore_index=ignore_index)
            )
            self.weights.append(gstmse_weight)

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(
        self, preds: torch.Tensor, gts: torch.Tensor, sim_index: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            preds: torch.float (N, C, T).
            gts: torch.long (N, T).
            sim_index: torch.float (N, C', T).
        """

        loss = 0.0
        for criterion, weight in zip(self.criterions, self.weights):
            if isinstance(criterion, GaussianSimilarityTMSE):
                loss += weight * criterion(preds, gts, sim_index) #高斯平滑损失
            else:
                loss += weight * criterion(preds, gts)#交叉熵

        return loss


class BoundaryRegressionLoss(nn.Module):
    """
    Boundary Regression Loss
        bce: Binary Cross Entropy Loss for Boundary Prediction
        mse: Mean Squared Error
    """

    def __init__(
        self,
        bce: bool = True,
        focal: bool = False,
        mse: bool = False,
        weight: Optional[float] = None,
        pos_weight: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.criterions = []

        if bce:
            self.criterions.append(
                nn.BCEWithLogitsLoss(weight=weight, pos_weight=pos_weight)
            )

        if focal:
            self.criterions.append(FocalLoss())

        if mse:
            self.criterions.append(nn.MSELoss())

        if len(self.criterions) == 0:
            print("You have to choose at least one loss function.")
            sys.exit(1)

    def forward(self, preds: torch.Tensor, gts: torch.Tensor, masks: torch.Tensor):
        """
        Args:
            preds: torch.float (N, 1, T).
            gts: torch. (N, 1, T).
            masks: torch.bool (N, 1, T).
        """
        loss = 0.0
        batch_size = float(preds.shape[0])

        for criterion in self.criterions:
            for pred, gt, mask in zip(preds, gts, masks):
                loss += criterion(pred[mask], gt[mask])

        return loss / batch_size


class KLLoss(nn.Module):
    """Loss that uses a 'hinge' on the lower bound.
    This means that for samples with a label value smaller than the threshold, the loss is zero if the prediction is
    also smaller than that threshold.
    args:
        error_matric:  What base loss to use (MSE by default).
        threshold:  Threshold to use for the hinge.
        clip:  Clip the loss if it is above this value.
    """

    def __init__(self, reduction = 'mean'):
        super().__init__()
        print('=========using KL Loss=and has temperature and * bz==========')
        self.error_metric = nn.KLDivLoss(reduction=reduction) # 初始化 KL 散度损失函数，size_average 和 reduce 参数控制是否对损失进行平均和是否进行降维

    def forward(self, prediction, label):
        batch_size = prediction.shape[0] # 获取批次大小
        probs1 = F.log_softmax(prediction, 1) # 对模型的预测结果和标签进行 softmax 操作再log
        probs2 = F.softmax(label * 10, 1) # 在标签上应用温度调节
        loss = self.error_metric(probs1, probs2) * batch_size # 计算 KL 散度损失，乘以批次大小进行标量化
        return loss


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self):
        # ignore_index = -100, reduction = 'mean'
        super(LabelSmoothingCrossEntropy, self).__init__()
    def forward(self, x, smoothed_target,_mask):
        logprobs = F.log_softmax(x, dim=-1)
        loss = - logprobs * smoothed_target * _mask
        loss = loss[torch.nonzero(loss,as_tuple=True)].mean()
        return loss

class CosSimLoss(nn.Module):
    def __init__(self): # I'm assuming the embedding matrices are same sizes.
        super(CosSimLoss, self).__init__()
        self.cosine = nn.CosineSimilarity()
        self.sigmoid = nn.Sigmoid()
        self.bce = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, a, b):
        # a: 5 * N * C b: 5 * N * C
        bz, num, c = a.shape
        bz_list = []
        for idx in range(bz):
            similarity = torch.matmul(a[idx], b[idx].T)
            loss = self.bce(similarity, torch.zeros(num,num))
            bz_list.append(loss)
        _loss = torch.stack(bz_list)
        return _loss

class SegCrossEntropyLoss(nn.Module):
    def __init__(self, reweight=False, thresold = 0.9, ignore_index=255,return_prob=False):
        super(SegCrossEntropyLoss, self).__init__()
        self.ce = nn.CrossEntropyLoss(ignore_index=ignore_index,reduction='none')
        self.reweight = reweight
        self.thresold = thresold
        self.return_prob = return_prob
    def forward(self, pred, target):
        # pred: N C T target: N T
        loss = self.ce(pred, target)
        if self.reweight:
            prob = torch.softmax(pred, dim=1)
            prob = torch.gather(prob, 1, target.unsqueeze(1).expand(-1, prob.size(1), -1))
            loss = loss[prob<self.thresold]
        loss = loss.mean()
        if self.return_prob:
            return loss, prob
        else:
            return loss


class SCELoss(nn.Module):
    def __init__(self, logit_scale=None):
        super(SCELoss, self).__init__()
        # self.logit_scale = np.log(1 / 0.07)
    def forward(self, parts, centers):
        K = parts.size(0)
        loss = 0
        for e in range(K):
            logits = parts[e] @ centers.T
            loss += torch.log(torch.exp(logits[e]) / torch.exp(logits).sum())

        loss = -loss / K
        return loss