
import os
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn.functional as F
from utils.class_id_map import get_id2class_map
from utils.config import Config
from utils.metric import AverageMeter, BoundaryScoreMeter, ScoreMeter
from utils.postprocess import PostProcessor
from tqdm import tqdm
from utils.common_tools import segment_video_labels, gen_label, generate_segment_features, create_logits
from TextEncoder.Clip import Model as TextModel

def train(
        train_loader: DataLoader,
        model: nn.Module,
        criterion_cls: nn.Module,
        criterion_bound: nn.Module,
        criterion_contrast: nn.Module,
        optimizer: optim.Optimizer,
        **kwargs
) -> float:
    losses = AverageMeter("Loss", ":.4e")
    losses2 = AverageMeter("Loss2", ":.4e")
    model.train()
    model_text: TextModel = kwargs['model_text']
    model_text.soa.train()
    config: Config = kwargs['config']
    opt2 = kwargs['opt2']
    device = config.device
    lambda_bound_loss = config.lambda_b
    for sample in tqdm(train_loader):
        x = sample["feature"].to(device)
        t = sample["label"].to(device)
        b = sample["boundary"].to(device)
        mask = sample["mask"].to(device)
        optimizer.zero_grad()
        opt2.zero_grad()
        batch_size = x.shape[0]
        output_cls, output_bound, proj_feature_list, logit_scale = model(x, mask)
        t_segment = segment_video_labels(t, ignore_idx=config.none_idx)
        label = [i[0] for seg in t_segment for i in seg]

        loss = 0.0
        if isinstance(output_cls, list):
            n = len(output_cls)
            for out in output_cls:
                loss += criterion_cls(out, t, x) / n
        else:
            loss += criterion_cls(output_cls, t, x)

        if isinstance(output_bound, list):
            n = len(output_bound)
            for out in output_bound:
                loss += lambda_bound_loss * criterion_bound(out, b, mask) / n
        else:
            loss += lambda_bound_loss * criterion_bound(output_bound, b, mask)
        label_g = torch.from_numpy(gen_label(label)).long()
        contra_loss = 0.0
        _label, _count = np.unique(np.array(label), return_counts=True)
        _count = _count.tolist()
        __count = []
        for i in range(config.n_classes):
            if i not in _label:
                __count.append(0)
            else:
                __count.append(_count.pop(0))
        _weight = 1 / np.array(__count)
        weight = torch.tensor(_weight[label]).to(device)
        part_num = len(config.part_text_idx)
        for ind in range(part_num):
            action_embedding = generate_segment_features(proj_feature_list[ind], t_segment, device)
            text_embedding = model_text(part_id=config.part_text_idx[ind], class_ids=label)
            logits_per_image, _ = create_logits(action_embedding, text_embedding, 1 if config.dataset == 'LARA' else logit_scale[:, ind].mean())
            contra_loss += (criterion_contrast(logits_per_image, label_g.float().to(device)).mean(dim=1)* weight).mean()
        loss += contra_loss
        losses2.update(contra_loss.item(), batch_size)
        losses.update(loss.item(), batch_size)  # 记录loss
        loss.backward()
        optimizer.step()
        opt2.step()

    return losses.avg, losses2.avg


def validate(
        val_loader: DataLoader,
        model: nn.Module,
        criterion_cls: nn.Module,
        criterion_bound: nn.Module,
        config
) -> Tuple[float, float, float, float, float, float, float, float, str]:
    losses = AverageMeter("Loss", ":.4e")
    lambda_b = config.lambda_b
    dataset = config.dataset
    dataset_dir = config.dataset_dir
    iou_thresholds = config.iou_thresholds
    boundary_th = config.boundary_th
    tolerance = config.tolerance
    refinement_method = config.refinement_method
    postprocessor = PostProcessor(refinement_method, boundary_th)
    device = config.device
    scores_cls = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output_cls, output_bound = model(x, mask)  # 模型计算边界以及计算分类结果

            loss = 0.0
            loss += criterion_cls(output_cls, t, x)
            loss += lambda_b * criterion_bound(output_bound, b, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()  # 将结果变成ndarray

            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()  # groundtruth of 分类和标签
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )  # 加上了边界的预测
            # update score
            scores_cls.update(output_cls, t, output_bound, mask)  # 经典acc,edit以及tp，fn，fp，tn
            scores_bound.update(output_bound, b, mask)  # tp，fn，fp，tn
            scores_after_refinement.update(refined_output_cls, t)  # 经典acc,edit以及tp，fn，fp，tn

    cls_acc, edit_score, segment_f1s = scores_after_refinement.get_scores()
    bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

    return (
        losses.avg,
        cls_acc,
        edit_score,
        segment_f1s
    )


def test(
        val_loader: DataLoader,
        model: nn.Module,
        criterion_cls: nn.Module,
        criterion_bound: nn.Module,
        config
) -> Tuple[float, float, float, float, float, float, float, float, str]:
    losses = AverageMeter("Loss", ":.4e")
    lambda_b = config.lambda_b
    dataset = config.dataset
    dataset_dir = config.dataset_dir
    iou_thresholds = config.iou_thresholds
    boundary_th = config.boundary_th
    tolerance = config.tolerance
    refinement_method = config.refinement_method
    postprocessor = PostProcessor(refinement_method, boundary_th)
    device = config.device
    scores_cls = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )
    scores_bound = BoundaryScoreMeter(
        tolerance=tolerance, boundary_threshold=boundary_th
    )

    scores_after_refinement = ScoreMeter(
        id2class_map=get_id2class_map(dataset, dataset_dir=dataset_dir),
        iou_thresholds=iou_thresholds,
    )

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        for sample in tqdm(val_loader):
            x = sample["feature"]
            t = sample["label"]
            b = sample["boundary"]
            mask = sample["mask"]

            x = x.to(device)
            t = t.to(device)
            b = b.to(device)
            mask = mask.to(device)

            batch_size = x.shape[0]

            # compute output and loss
            output_cls, output_bound = model(x, mask)  # 模型计算边界以及计算分类结果

            loss = 0.0
            loss += criterion_cls(output_cls, t, x)
            loss += lambda_b * criterion_bound(output_bound, b, mask)

            # measure accuracy and record loss
            losses.update(loss.item(), batch_size)

            # calcualte accuracy and f1 score
            output_cls = output_cls.to("cpu").data.numpy()
            output_bound = output_bound.to("cpu").data.numpy()  # 将结果变成ndarray

            t = t.to("cpu").data.numpy()
            b = b.to("cpu").data.numpy()  # groundtruth of 分类和标签
            mask = mask.to("cpu").data.numpy()

            refined_output_cls = postprocessor(
                output_cls, boundaries=output_bound, masks=mask
            )  # 加上了边界的预测
            # update score
            scores_cls.update(output_cls, t, output_bound, mask)  # 经典acc,edit以及tp，fn，fp，tn
            scores_bound.update(output_bound, b, mask)  # tp，fn，fp，tn
            scores_after_refinement.update(refined_output_cls, t)  # 经典acc,edit以及tp，fn，fp，tn

    cls_acc, edit_score, segment_f1s = scores_after_refinement.get_scores()
    bound_acc, precision, recall, bound_f1s = scores_bound.get_scores()

    return (
        losses.avg,
        cls_acc,
        edit_score,
        segment_f1s
    )
