import sys

import numpy as np
import torch
import torch.nn as nn
import math
import math

import torch
import torch.nn as nn

def import_class(class_name):
    module = __import__(class_name.rsplit('.',1)[0], fromlist=[class_name.rsplit('.',1)[1]])
    return getattr(module, class_name.rsplit('.',1)[1])

def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p*idx_decoder)

def change_label_score(best_test, train_loss, epoch, cls_acc, edit_score, f1s):

    best_test['train_loss'] = train_loss
    best_test['epoch'] = epoch
    best_test['cls_acc'] = cls_acc
    best_test['edit'] = edit_score
    best_test['f1s@0.1'] = f1s[0]
    best_test['f1s@0.25'] = f1s[1]
    best_test['f1s@0.5'] = f1s[2]

def generate_segment_features(output_feature, t_segment):
    segment_features = []

    for i in range(len(t_segment)):
        segment_list = t_segment[i]

        for j in range(len(segment_list)):
            start_frame = segment_list[j][2]
            end_frame = segment_list[j][3] + 1

            # 提取每一段的帧级特征，并计算平均值
            segment_feature = output_feature[i, :, start_frame:end_frame].mean(dim=2)

            # 添加到段级特征列表中
            segment_features.append(segment_feature)

    # 将段级特征列表转换为张量
    segment_features = torch.stack(segment_features)

    return segment_features


def segment_video_labels(video_labels_batch, ignore_idx):
    segments_batch = []

    for video_labels in video_labels_batch:
        segments = []
        current_label = None
        segment_start = 0
        for i, label in enumerate(video_labels):
            label = label.item()
            if label == ignore_idx:
                continue
            if label != current_label:
                if (current_label is not None) and (label !=255):
                    segment_length = i - segment_start
                    segments.append((current_label, segment_length, segment_start, i - 1))
                current_label = label
                segment_start = i

        # 处理最后一个段
        if (current_label is not None) and (video_labels[-1] !=255) and (video_labels[-1] !=0):
            segment_length = len(video_labels) - segment_start
            segments.append((current_label, segment_length, segment_start, len(video_labels) - 1))
        segments_batch.append(segments)

    return segments_batch


def gen_label(labels):
    num = len(labels)
    gt = np.zeros(shape=(num,num)) #（N，N）
    for i, label in enumerate(labels):
        for k in range(num):
            if labels[k] == label: #把同样都是这个类的都置为1
                gt[i,k] = 1
    return gt #对比矩阵的gt


def generate_segment_features(output_feature, t_segment, device):
    segment_features = []

    for i in range(len(t_segment)):
        segment_list = t_segment[i]

        for j in range(len(segment_list)):
            start_frame = segment_list[j][2]
            end_frame = segment_list[j][3] + 1

            # 提取每一段的帧级特征，并计算平均值
            segment_feature = output_feature[i, :, start_frame:end_frame].mean(dim=-1)

            # 添加到段级特征列表中
            segment_features.append(segment_feature)

    # 将段级特征列表转换为张量
    segment_features = torch.stack(segment_features).to(device)

    return segment_features

def create_logits(x1, x2, logit_scale):
    x1 = x1 / x1.norm(dim=-1, keepdim=True) #归一化为模长为1 在512维度上
    x2 = x2 / x2.norm(dim=-1, keepdim=True) #归一化为模长为1

    # cosine similarity as logits
    logits_per_x1 = logit_scale * x1 @ x2.t() #余弦相似度 模为1,所以直接矩阵相乘就可以
    logits_per_x2 = logit_scale * x2 @ x1.t()

    # shape = [global_batch_size, global_batch_size]
    return logits_per_x1, logits_per_x2 #两个分别是x1对x2的预先相似度与反之

def create_logits_with_class_logit_scale(x1, x2, logit_scales):
    # logit_scales: n_classes-1 * n_classes-1
    # label_g: k
    x1 = x1 / x1.norm(dim=-1, keepdim=True) #归一化为模长为1 在512维度上
    x2 = x2 / x2.norm(dim=-1, keepdim=True) #归一化为模长为1
    _logit_scales = logit_scales[label_g]
    # cosine similarity as logits
    logits = logit_scale * x1 @ x2.t() #余弦相似度 模为1,所以直接矩阵相乘就可以

    # shape = [global_batch_size, global_batch_size]
    return logits #两个分别是x1对x2的预先相似度与反之