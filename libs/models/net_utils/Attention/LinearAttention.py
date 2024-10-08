import torch
import torch.nn as nn


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


class LinearAtt(nn.Module):
    def __init__(self, dilation, in_channel, out_channel, stage, alpha):
        super(LinearAtt, self).__init__()
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