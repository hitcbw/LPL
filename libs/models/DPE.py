from typing import Any, Optional, Tuple

import torch
import torch.nn as nn
import copy
import math
import numpy as np

from utils.config import Config
from .tcn import SingleStageTCN
from .SP import MultiScale_GraphConv


def exponential_descrease(idx_decoder, p=3):
    return math.exp(-p * idx_decoder)


class Linear_Attention(nn.Module):
    def __init__(self,
                 in_channel,
                 n_features,
                 out_channel,
                 n_heads=4,
                 drop_out=0.05
                 ):
        super().__init__()
        self.n_heads = n_heads

        self.query_projection = nn.Linear(in_channel, n_features)
        self.key_projection = nn.Linear(in_channel, n_features)
        self.value_projection = nn.Linear(in_channel, n_features)
        self.out_projection = nn.Linear(n_features, out_channel)  # 都走64-64的线性层
        self.dropout = nn.Dropout(drop_out)  # 0.05的dropout

    def elu(self, x):
        return torch.sigmoid(x)
        # return torch.nn.functional.elu(x) + 1

    def forward(self, queries, keys, values, mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        queries = self.query_projection(queries).view(B, L, self.n_heads, -1)
        keys = self.key_projection(keys).view(B, S, self.n_heads, -1)
        values = self.value_projection(values).view(B, S, self.n_heads, -1)

        queries = queries.transpose(1, 2)
        keys = keys.transpose(1, 2)
        values = values.transpose(1, 2)  # （n,head,t,c）

        queries = self.elu(queries)
        keys = self.elu(keys)
        KV = torch.einsum('...sd,...se->...de', keys, values)  # （n,head,t,c）,（n,head,t,c）->(n,head,c,c)
        Z = 1.0 / torch.einsum('...sd,...d->...s', queries,
                               keys.sum(dim=-2) + 1e-6)  # （n,head,t,c）,（n,head,c） ->(n,head,t) 一个全局的，即q的t个帧查k一个全局的信息

        x = torch.einsum('...de,...sd,...s->...se', KV, queries, Z).transpose(1,
                                                                              2)  # (n,head,c,c),(n,head,t,c),(n,head,t)->(n,head,t,c) 这不是通道注意力+一个局部注意全局的t的注意力

        x = x.reshape(B, L, -1)  # 直接4个头融合（n,t,c）
        x = self.out_projection(x)  # 线性层
        x = self.dropout(x)  # 0.05的dropout

        return x * mask[:, 0, :, None]


class AttModule(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(AttModule, self).__init__()
        self.stage = stage
        self.alpha = alpha

        self.feed_forward = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 3, padding=dilation, dilation=dilation),
            nn.ReLU()
        )  # 膨胀卷积
        self.instance_norm = nn.InstanceNorm1d(out_channel, track_running_stats=False)  # 这个应该改成batchnorm
        self.att_layer = Linear_Attention(out_channel, out_channel, out_channel)

        self.conv_out = nn.Conv1d(out_channel, out_channel, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, f, mask):

        out = self.feed_forward(x)  # 一个时间卷积
        if self.stage == 'encoder':
            q = self.instance_norm(out).permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, q, mask).permute(0, 2, 1) + out
        else:
            assert f is not None
            q = self.instance_norm(out).permute(0, 2, 1)
            f = f.permute(0, 2, 1)
            out = self.alpha * self.att_layer(q, q, f, mask).permute(0, 2, 1) + out  # 线性transformer

        out = self.conv_out(out)
        out = self.dropout(out)

        return (x + out) * mask


class SFI(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1)  # 19->64的卷积
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features))  # 64—>64的两层前馈

    def forward(self, feature_s, feature_t, mask):  # feature_s来自空间（n,t,v) feature_t来自上一层时间(n,t,c)
        feature_s = feature_s.permute(0, 2, 1)  # (n,v,t)
        n, c, t = feature_s.shape
        feature_s = self.conv_s(feature_s)  # (n,v,t)->(n,c,t)
        map = self.softmax(torch.einsum("nct,ndt->ncd", feature_s, feature_t) / t)  # （n,c,c）的图
        feature_cross = torch.einsum("ncd,ndt->nct", map, feature_t)  # 图（n,c,c)与（n,c,t）变成（n,c,t）
        feature_cross = feature_cross + feature_t  # 加上时间特征
        feature_cross = feature_cross.permute(0, 2, 1)  # (n,t,c）
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t  # 一个前馈层+残差

        return feature_cross * mask


class STI(nn.Module):
    def __init__(self, node, in_channel, n_features, out_channel, num_layers, SFI_layer, channel_masking_rate=0.3,
                 alpha=1):
        super().__init__()
        self.SFI_layer = SFI_layer  # （1,2,3,4,5,6,7,8,9）
        num_SFI_layers = len(SFI_layer)  # 9
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate)  # 0.3的通道dropout

        self.conv_in = nn.Conv2d(in_channel, num_SFI_layers + 1, kernel_size=1)  # 64->10的卷积
        self.conv_t = nn.Conv1d(node, n_features, 1)  # V=19->64的卷积
        self.SFI_layers = nn.ModuleList(
            [SFI(node, n_features) for i in range(num_SFI_layers)])  # 9层的，也就是针对于每个V=19->64的卷积+前馈
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'encoder', alpha) for i in
             range(num_layers)])  # 10层扩张注意力
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)  # 这个是输出层

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = self.dropout(x)  # 随机丢掉0.3

        count = 0
        x = self.conv_in(x)  # c=64->10
        feature_s, feature_t = torch.split(x, (len(self.SFI_layers), 1), dim=1)  # （n,10,t,v）->(n,9,t,v)+(n,1,t,v)
        feature_t = feature_t.squeeze(1).permute(0, 2, 1)  # (n,v,t)
        feature_st = self.conv_t(feature_t)  # (n,v,t)->(n,64,t)

        for index, layer in enumerate(self.layers):  # 10层时空融合（空间有个v-c的）以及时间膨胀卷积
            if index in self.SFI_layer:
                feature_st = self.SFI_layers[count](feature_s[:, count, :], feature_st, mask)
                count += 1
            feature_st = layer(feature_st, None, mask)  # 普通的膨胀卷积

        feature_st = self.conv_out(feature_st)
        return feature_st * mask


class CPAtt(nn.Module):
    def __init__(self, in_channel, n_features):
        super().__init__()
        self.conv_s = nn.Conv1d(in_channel, n_features, 1)  # 19->64的卷积
        self.softmax = nn.Softmax(dim=-1)
        self.ff = nn.Sequential(nn.Linear(n_features, n_features),
                                nn.GELU(),
                                nn.Dropout(0.3),
                                nn.Linear(n_features, n_features))  # 64—>64的两层前馈

    def forward(self, feature_s, feature_t, mask):  # feature_s来自空间（n,t,v) feature_t来自上一层时间(n,t,c)
        n, c, t = feature_s.shape
        map = self.softmax(torch.einsum("nct,ndt->ncd", feature_s, feature_t) / t)  # （n,c,c）的图
        feature_cross = torch.einsum("ncd,ndt->nct", map, feature_t)  # 图（n,c,c)与（n,c,t）变成（n,c,t）
        feature_cross = feature_cross + feature_t  # 加上时间特征
        feature_cross = feature_cross.permute(0, 2, 1)  # (n,t,c）
        feature_cross = self.ff(feature_cross).permute(0, 2, 1) + feature_t  # 一个前馈层+残差

        return feature_cross * mask


class DPFL(nn.Module):
    def __init__(self, node, in_channel, n_features, out_channel, num_layers, SFI_layer, head_list, hand_list,
                 foot_list, hip_list, channel_masking_rate=0.3, alpha=1):
        super().__init__()
        self.SFI_layer = SFI_layer  # （1,2,3,4,5,6,7,8,9）
        num_SFI_layers = len(SFI_layer)  # 9
        self.channel_masking_rate = channel_masking_rate
        self.dropout = nn.Dropout2d(p=channel_masking_rate)  # 0.3的通道dropout

        self.conv_in = nn.Conv2d(in_channel, 10, kernel_size=1)  # 64->10的卷积
        self.conv_t = nn.Conv2d(n_features, n_features, 1)  # V=19->64的卷积
        self.SFI_layers_list = nn.ModuleList()
        self.layers_list = nn.ModuleList()
        self.CPAtt_layers_list = nn.ModuleList()
        self.part_list = nn.ModuleList()
        self.conv_out = nn.ModuleList()
        head_list = torch.Tensor(head_list).long()
        hand_list = torch.Tensor(hand_list).long()
        foot_list = torch.Tensor(foot_list).long()
        hip_list = torch.Tensor(hip_list).long()
        global_list = torch.arange(node).long()
        node_list = [len(head_list), len(hand_list), len(hip_list), len(foot_list), len(global_list)]
        for idx in range(5):
            self.part_list.append(nn.Conv1d(node_list[idx], n_features, 1))  # the first  convolution layer
            self.SFI_layers_list.append(nn.ModuleList([SFI(node_list[idx], n_features) for _ in range(
                num_SFI_layers)]))  # part-level spatial-temporal cross-att layer
            self.layers_list.append(nn.ModuleList([AttModule(2 ** i, n_features, n_features, 'encoder', alpha) for i in
                                                   range(num_layers)]))  # part-level temporal layer
            if idx != 4:
                self.CPAtt_layers_list.append(nn.ModuleList(
                    [CPAtt(n_features, n_features) for _ in range(num_SFI_layers)]))  # cross-part att layer
            else:
                self.conv_out = nn.Conv1d(n_features * 5, out_channel, 1)  # global projection

        self.node_idx_list = [head_list, hand_list, hip_list, foot_list, global_list]

    def forward(self, x, mask):
        if self.channel_masking_rate > 0:
            x = self.dropout(x)

        count = 0
        N, C, T, V = x.shape
        x = self.conv_in(x) * mask.unsqueeze(-1)  # c=64->10
        feature_s, feature_t = torch.split(x, (len(self.SFI_layer), 1), dim=1)  # N 9 t v   N 1 t v
        # partion
        feature_list = [None for _ in range(5)]
        for i in range(5):
            feature_list[i] = feature_t[:, :, :, self.node_idx_list[i]].squeeze(1).permute(0, 2, 1).contiguous()
            feature_list[i] = self.part_list[i](feature_list[i]) * mask

        for index in range(len(self.SFI_layer) + 1):
            if index in self.SFI_layer:
                for i in range(5):
                    part_feature = feature_s[:, count, :, self.node_idx_list[i]]  # n t v
                    feature_list[i] = self.SFI_layers_list[i][count](part_feature, feature_list[i], mask)
                    if i != 4:
                        feature_list[i] = self.CPAtt_layers_list[i][count](feature_list[4], feature_list[i], mask)
                count += 1
            for i in range(5):
                feature_list[i] = self.layers_list[i][index](feature_list[i], None, mask)  # 普通的膨胀卷积
        fuse_feature = torch.cat(feature_list, dim=1) * mask
        feature_list[-1] = self.conv_out(fuse_feature) * mask
        return feature_list  # [head_feature, hand_feature, foot_feature, global_feature, fuse_feature]


class Decoder(nn.Module):
    def __init__(self, in_channel, n_features, out_channel, num_layers, alpha=1):
        super().__init__()

        self.conv_in = nn.Conv1d(in_channel, n_features, 1)
        self.layers = nn.ModuleList(
            [AttModule(2 ** i, n_features, n_features, 'decoder', alpha) for i in
             range(num_layers)])
        self.conv_out = nn.Conv1d(n_features, out_channel, 1)

    def forward(self, x, fencoder, mask):
        feature = self.conv_in(x)
        for layer in self.layers:
            feature = layer(feature, fencoder, mask)
        out = self.conv_out(feature)

        return out, feature


class Model(nn.Module):
    """
    this model predicts both frame-level classes and boundaries.
    Args:
        in_channel:
        n_feature: 64
        n_classes: the number of action classes
        n_layers: 10
    """

    def __init__(self, config: Config) -> None:
        super().__init__()
        in_channel = config.in_channel
        n_features = config.n_features
        n_classes = config.n_classes
        n_stages = config.n_stages
        n_layers = config.n_layers
        n_refine_layers = config.n_refine_layers
        n_stages_asb = config.n_stages_asb
        n_stages_brb = config.n_stages_brb
        SFI_layer = config.SFI_layer
        dataset = config.dataset
        node = config.n_node
        if not isinstance(n_stages_asb, int):
            n_stages_asb = n_stages

        if not isinstance(n_stages_brb, int):
            n_stages_brb = n_stages

        super().__init__()

        self.logit_scale = nn.Parameter(torch.ones(1, 5) * np.log(1 / 0.07))  # 2.6593

        self.in_channel = in_channel

        self.SP = MultiScale_GraphConv(13, in_channel, n_features, dataset,n_node=node)  # 多跳图卷积
        self.STI = DPFL(node, n_features, n_features, n_features, num_layers=n_layers, SFI_layer=SFI_layer,
                        head_list=config.head_list, hand_list=config.hand_list, foot_list=config.foot_list,
                        hip_list=config.hip_list)

        self.conv_cls = nn.Conv1d(n_features, n_classes, 1)  # 分类头
        self.conv_bound = nn.Conv1d(n_features, 1, 1)  # 回归头
        asb = [
            copy.deepcopy(Decoder(n_classes, n_features, n_classes, n_refine_layers, alpha=exponential_descrease(s)))
            for s in range(n_stages_asb - 1)
        ]
        self.conv_asb_feature = nn.Conv1d(n_features, n_features, 1)
        brb = [
            SingleStageTCN(1, n_features, 1, n_refine_layers) for _ in range(n_stages_brb - 1)
        ]
        self.asb = nn.ModuleList(asb)
        self.brb = nn.ModuleList(brb)

        self.activation_asb = nn.Softmax(dim=1)
        self.activation_brb = nn.Sigmoid()
        self.part_list = nn.ModuleList()

        for i in range(5):
            self.part_list.append(nn.Conv1d(n_features, config.n_contra_features, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        x = self.SP(x) * mask.unsqueeze(3)
        features = self.STI(x, mask)
        feature = features[-1]
        out_cls = self.conv_cls(feature)* mask
        out_bound = self.conv_bound(feature)* mask
        proj_features = [self.part_list[i](features[i]) * mask for i in range(5)]
        if self.training:
            outputs_cls = [out_cls]
            outputs_bound = [out_bound]

            for as_stage in self.asb:
                out_cls, feature = as_stage(self.activation_asb(out_cls) * mask, feature * mask, mask)
                outputs_cls.append(out_cls)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask)
                outputs_bound.append(out_bound)

            return outputs_cls, outputs_bound, proj_features, self.logit_scale
        else:
            for as_stage in self.asb:
                out_cls, _ = as_stage(self.activation_asb(out_cls) * mask, feature * mask, mask)

            for br_stage in self.brb:
                out_bound = br_stage(self.activation_brb(out_bound), mask)

            return out_cls, out_bound
