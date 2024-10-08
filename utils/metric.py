# https://github.com/cthincsl/TemporalConvthutionalNetworks/blob/master/code/metrics.py
# Score metric for action segmentation was originally written by cthincs1

import copy
import csv
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def get_segments( #把帧级标签变为段级标签
    frame_wise_label: np.ndarray,
    id2class_map: Dict[int, str],
    bg_class: str = "background",
) -> Tuple[List[int], List[int], List[int]]:
    """
    Args:
        frame-wise label: frame-wise prediction or ground truth. 1D numpy array
    Return:
        segment-label array: list (excluding background class)
        start index list
        end index list
    """

    labels = []
    starts = []
    ends = [] #创建三个标签，分别存储动作片段的标签和前后index

    frame_wise_label = [
        id2class_map[frame_wise_label[i]] for i in range(len(frame_wise_label))
    ]

    # get class, start index and end index of segments
    # background class is excluded
    last_label = frame_wise_label[0]
    if frame_wise_label[0] != bg_class:
        labels.append(frame_wise_label[0])
        starts.append(0)

    for i in range(len(frame_wise_label)):
        # if action labels change
        if frame_wise_label[i] != last_label: #如果标签不同（到下一段了）
            # if label change from one class to another class
            # it's an action starting point
            if frame_wise_label[i] != bg_class:  #如果不是背景，就接着加段级信息
                labels.append(frame_wise_label[i]) #标签类别存入
                starts.append(i) #此段的开始index存入

            # if label change from background to a class
            # it's not an action end point.
            if last_label != bg_class: #如果上一段标签不是背景帧
                ends.append(i) #存入结束index

            # update last label
            last_label = frame_wise_label[i] #更新上一段的标签

    if last_label != bg_class: #如果最后一帧也不是背景
        ends.append(i) #加上结束index

    return labels, starts, ends


def levenshtein(pred: List[int], gt: List[int], norm: bool = True) -> float:
    """
    Levenshtein distance(Edit Distance)
    Args:
        pred: segments list
        gt: segments list
    Return:
        if norm == True:
            (1 - average_edit_distance) * 100
        else:
            edit distance
    """

    n, m = len(pred), len(gt) #先看预测和gt各有多少长度

    dp = [[0] * (m + 1) for _ in range(n + 1)] #（pre段数+1）个list，每个list里（gt段数+1）个0
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j #相当于加表头

    for i in range(1, n + 1): #开始给表格填数
        for j in range(1, m + 1):
            cost = 0 if pred[i - 1] == gt[j - 1] else 1 #如果第i个pre的标签和第j个gt的标签相等，cost=0,否则1
            dp[i][j] = min(
                dp[i - 1][j] + 1,  # insertion
                dp[i][j - 1] + 1,  # deletion
                dp[i - 1][j - 1] + cost,
            )  # replacement  #计算编辑分数

    if norm:
        score = (1 - dp[n][m] / max(n, m)) * 100 #计算百分制的编辑分数
    else:
        score = dp[n][m]

    return score


def get_n_samples(
    p_label: List[int],
    p_start: List[int],
    p_end: List[int],
    g_label: List[int],
    g_start: List[int],
    g_end: List[int],
    iou_threshold: float,
    bg_class: List[str] = ["background"],
) -> Tuple[int, int, int]:
    """
    Args:
        p_label, p_start, p_end: return values of get_segments(pred)
        g_label, g_start, g_end: return values of get_segments(gt)
        threshold: threshold (0.1, 0.25, 0.5)
        bg_class: background class
    Return:
        tp: true positive
        fp: false positve
        fn: false negative
    """

    tp = 0
    fp = 0
    hits = np.zeros(len(g_label))

    for j in range(len(p_label)):
        intersection = np.minimum(p_end[j], g_end) - np.maximum(p_start[j], g_start)
        union = np.maximum(p_end[j], g_end) - np.minimum(p_start[j], g_start)
        IoU = (1.0 * intersection / union) * (
            [p_label[j] == g_label[x] for x in range(len(g_label))]
        )
        # Get the best scoring segment
        idx = np.array(IoU).argmax()

        if IoU[idx] >= iou_threshold and not hits[idx]:
            tp += 1
            hits[idx] = 1
        else:
            fp += 1

    fn = len(g_label) - sum(hits)

    return float(tp), float(fp), float(fn)


class ScoreMeter(object):
    def __init__(
        self,
        id2class_map: Dict[int, str],
        iou_thresholds: Tuple[float] = (0.1, 0.25, 0.5),
        ignore_index: int = 255,
    ) -> None:

        self.iou_thresholds = iou_thresholds  # threshold for f score(0.1, 0.25, 0.5, 0.75, 0.9)
        self.ignore_index = ignore_index #255
        self.id2class_map = id2class_map #map
        self.edit_score = 0
        self.tp = [0 for _ in range(len(iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.n_classes = len(self.id2class_map)
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    def _fast_hist(self, pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
        mask = (gt >= 0) & (gt < self.n_classes) #看看都有没有效，有效mask就是true
        hist = np.bincount(
            self.n_classes * gt[mask].astype(int) + pred[mask], #这个还真是高级  gt相当于每行，pred相当于每行的12个元素，就像是十位和个位一样
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes) #计算混淆矩阵的
        return hist

    def update(
        self,
        outputs: np.ndarray,
        gts: np.ndarray,
        boundaries: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
    ) -> None:
        """
        Args:
            outputs: np.array. shape(N, C, T)
                the model output for boundary prediciton
            gt: np.array. shape(N, T)
                Ground Truth for boundary
        """
        if len(outputs.shape) == 3:
            preds = outputs.argmax(axis=1) #变成索引（1,6000)
        elif len(outputs.shape) == 2:
            preds = copy.copy(outputs)

        for pred, gt in zip(preds, gts):
            pred = pred[gt != self.ignore_index]
            gt = gt[gt != self.ignore_index] #去掉该忽略的

            for lt, lp in zip(pred, gt):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten()) #把预测和gt输入，得出混淆矩阵

            self.n_videos += 1 #视频数量+1
            # count the correct frame
            self.n_frames += len(pred) #总帧数
            for i in range(len(pred)):
                if pred[i] == gt[i]:
                    self.n_correct += 1 #计算正确帧数

            # calculate the edit distance
            p_label, p_start, p_end = get_segments(pred, self.id2class_map) #拿到pred的段级标签（标签，起始，结尾）
            g_label, g_start, g_end = get_segments(gt, self.id2class_map) #同理，拿到gt的

            self.edit_score += levenshtein(p_label, g_label, norm=True) #计算编辑分数

            for i, th in enumerate(self.iou_thresholds): #跟据5种阈值计算f1-socre
                tp, fp, fn = get_n_samples(
                    p_label, p_start, p_end, g_label, g_start, g_end, th
                )
                self.tp[i] += tp
                self.fp[i] += fp
                self.fn[i] += fn

    def get_scores(self) -> Tuple[float, float, float]:
        """
        Return:
            Accuracy
            Normlized Edit Distance
            F1 Score of Each Threshold
        """

        # accuracy
        acc = 100 * float(self.n_correct) / self.n_frames #acc

        # edit distance
        edit_score = float(self.edit_score) / self.n_videos #edit分数除全视频数，得出每个视频的平均

        # F1 Score
        f1s = []
        for i in range(len(self.iou_thresholds)): #计算5个阈值的F1分数
            precision = self.tp[i] / float(self.tp[i] + self.fp[i])
            recall = self.tp[i] / float(self.tp[i] + self.fn[i])

            f1 = 2.0 * (precision * recall) / (precision + recall + 1e-7)
            f1 = np.nan_to_num(f1) * 100

            f1s.append(f1)

        # Accuracy, Edit Distance, F1 Score
        return acc, edit_score, f1s

    def return_confusion_matrix(self) -> np.ndarray:
        return self.confusion_matrix

    def save_scores(self, save_path: str) -> None:
        acc, edit_score, segment_f1s = self.get_scores()

        # save log
        columns = ["cls_acc", "edit"]
        data_dict = {
            "cls_acc": [acc],
            "edit": [edit_score],
        }

        for i in range(len(self.iou_thresholds)):
            key = "segment f1s@{}".format(self.iou_thresholds[i])
            columns.append(key)
            data_dict[key] = [segment_f1s[i]]

        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)

    def save_confusion_matrix(self, save_path: str) -> None:
        with open(save_path, "w") as file:
            writer = csv.writer(file, lineterminator="\n")
            writer.writerows(self.confusion_matrix)

    def reset(self) -> None:
        self.edit_score = 0
        self.tp = [0 for _ in range(len(self.iou_thresholds))]  # true positive
        self.fp = [0 for _ in range(len(self.iou_thresholds))]  # false positive
        self.fn = [0 for _ in range(len(self.iou_thresholds))]  # false negative
        self.n_correct = 0
        self.n_frames = 0
        self.n_videos = 0
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))


def argrelmax(prob: np.ndarray, threshold: float = 0.7) -> List[int]:
    """
    Calculate arguments of relative maxima.
    prob: np.array. boundary probability maps distributerd in [0, 1]
    prob shape is (T)
    ignore the peak whose value is under threshold

    Return:
        Index of peaks for each batch
    """
    # ignore the values under threshold
    prob[prob < threshold] = 0.0 #小于0.5的清零

    # calculate the relative maxima of boundary maps
    # treat the first frame as boundary
    peak = np.concatenate(
        [
            np.ones((1), dtype=bool), # 在数组开头添加一个 True，表示可能存在的第一个峰值
            (prob[:-2] < prob[1:-1]) & (prob[2:] < prob[1:-1]), # 检查了 prob 数组中每个元素是否大于其前后两个元素，如果是，则表示该位置为峰值
            np.zeros((1), dtype=bool), # 在数组结尾添加一个 False
        ],
        axis=0,
    ) #峰值的帧为true，其余为false

    peak_idx = np.where(peak)[0].tolist() #找到峰值帧索引

    return peak_idx #峰值帧索引


class BoundaryScoreMeter(object):
    def __init__(self, tolerance=5, boundary_threshold=0.7):
        # max distance of the frame which can be regarded as correct
        self.tolerance = tolerance #5

        # threshold of the boundary value which can be regarded as action boundary
        self.boundary_threshold = boundary_threshold #0.5
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0

    def update(self, preds, gts, masks):
        """
        Args:
            preds: np.array. the model output(N, T)
            gts: np.array. boudnary ground truth array (N, T)
            masks: np.array. np.bool. valid length for each video (N, T)
        Return:
            Accuracy
            Boundary F1 Score
        """

        for pred, gt, mask in zip(preds, gts, masks):
            # ignore invalid frames
            pred = pred[mask]
            gt = gt[mask] #gt和预测的有效部分

            pred_idx = argrelmax(pred, threshold=self.boundary_threshold)
            gt_idx = argrelmax(gt, threshold=self.boundary_threshold) #二者的边界索引

            n_frames = pred.shape[0]
            tp = 0.0
            fp = 0.0
            fn = 0.0

            hits = np.zeros(len(gt_idx))

            # calculate true positive, false negative, false postive, true negative
            for i in range(len(pred_idx)):
                dist = np.abs(np.array(gt_idx) - pred_idx[i])
                min_dist = np.min(dist) #找和当前预测边界最近的gt边界差距
                idx = np.argmin(dist) #找和当前预测边界最近的gt边界索引

                if min_dist <= self.tolerance and hits[idx] == 0: #如果差距在5内且没有被填过
                    tp += 1 #tp
                    hits[idx] = 1 #gt被用过了
                else:
                    fp += 1

            fn = len(gt_idx) - sum(hits)
            tn = n_frames - tp - fp - fn

            self.tp += tp
            self.fp += fp
            self.fn += fn
            self.n_frames += n_frames
            self.n_correct += tp + tn

    def get_scores(self):
        """
        Return:
            Accuracy
            Boundary F1 Score
        """

        # accuracy
        acc = 100 * self.n_correct / self.n_frames #acc

        # Boudnary F1 Score
        precision = self.tp / float(self.tp + self.fp)
        recall = self.tp / float(self.tp + self.fn)

        f1s = 2.0 * (precision * recall) / (precision + recall + 1e-7)
        f1s = np.nan_to_num(f1s) * 100

        # Accuracy, Edit Distance, F1 Score
        return acc, precision * 100, recall * 100, f1s

    def save_scores(self, save_path: str) -> None:
        acc, precision, recall, f1s = self.get_scores()

        # save log
        columns = ["bound_acc", "precision", "recall", "bound_f1s"]
        data_dict = {
            "bound_acc": [acc],
            "precision": [precision],
            "recall": [recall],
            "bound_f1s": [f1s],
        }

        df = pd.DataFrame(data_dict, columns=columns)
        df.to_csv(save_path, index=False)

    def reset(self):
        self.tp = 0.0  # true positive
        self.fp = 0.0  # false positive
        self.fn = 0.0  # false negative
        self.n_correct = 0.0
        self.n_frames = 0.0


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name: str, fmt: str = ":f") -> None:
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self) -> None:
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self) -> str:
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)
