import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import math
from math import sqrt
import os


class AutoCorrelation(nn.Module):
    """
    AutoCorrelation Mechanism with the following two phases:两个阶段自动关联机制:
    (1) period-based dependencies discovery基于周期的依赖关系发现
    (2) time delay aggregation时间延迟聚合
    This block can replace the self-attention family mechanism seamlessly.可以取代自注意力机制
    最后返回的是V
    """
    def __init__(self, mask_flag=True, factor=1, scale=None, attention_dropout=0.1, output_attention=False):
        super(AutoCorrelation, self).__init__()
        self.factor = factor
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def time_delay_agg_training(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)自相关的加速版本
        This is for the training phase.针对训练阶段
        """
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)   # 做大量的mean是为了得到下面的index
        #  通过得到的 corr 来计算出 权重最大几个index，从而获得权重值，根据 index 来对 values 进行移位得到滞后系列
        # 这个mean_value是怎么来的呢，是对这个corr（32，8，64，96）的8和64这个维度进行取mean，得到的是一个（32，96）的mean，
        # 这个32是batch_size，也就是说对这个batch中的每个corr都取平均。
        index = torch.topk(torch.mean(mean_value, dim=0), top_k, dim=-1)[1]

        # index最终的数是[0,94,1,95]也就是说对于这个96这个维度的数进行取top？得到其中corr比较大的索引
        weights = torch.stack([mean_value[:, index[i]] for i in range(top_k)], dim=-1)   # 将权重较大的attention进行堆叠
        # 得到（32，4）的向量，他是对原本的mean_value进行堆叠，
        # 所以weights的意义在于，把这个batch中的每个数据的这个比较大的数的维度的值都取出来进行堆叠，
        # 也就是说这个weights是这32个数据内部corr比较大的位置的数据的堆叠。这个weights也就相当于将比较活跃的attention进行提取出来堆叠。

        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)    # 这一步是对这个比较活跃的部分进行softmax

        # aggregation
        tmp_values = values   # 第一步先把原始的values给这个tmp_values，现在这个tmp_values和原来的values都是一样的都是维度为（32，8，64，96）维度的数据
        delays_agg = torch.zeros_like(values).float()   # 这一步生成一个和values一样维度的0向量，他的维度也是（32，8，64，96）
        for i in range(top_k):
            pattern = torch.roll(tmp_values, -int(index[i]), -1)    # 将-1这个维度像上移动index的这个步数。
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        #     然后用这个得到的pattern和经过softmax的比较大的那个类似qk的weights进行相乘。
        # 返回的是delays_agg是自相关模块后的结果 不过是相当于 Transformer 模块的输出而已
        return delays_agg    # 延迟聚合？？？是什么东西

    def time_delay_agg_inference(self, values, corr):
        """
        SpeedUp version of Autocorrelation (a batch-normalization style design)
        This is for the inference phase.   用于推断阶段
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        mean_value = torch.mean(torch.mean(corr, dim=1), dim=1)
        weights, delay = torch.topk(mean_value, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation聚合
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * \
                         (tmp_corr[:, i].unsqueeze(1).unsqueeze(1).unsqueeze(1).repeat(1, head, channel, length))
        return delays_agg

    def time_delay_agg_full(self, values, corr):
        """
        Standard version of Autocorrelation标准版本的自相关
        这个函数好像没有用到
        """
        batch = values.shape[0]
        head = values.shape[1]
        channel = values.shape[2]
        length = values.shape[3]
        # index init
        init_index = torch.arange(length).unsqueeze(0).unsqueeze(0).unsqueeze(0)\
            .repeat(batch, head, channel, 1).to(values.device)
        # find top k
        top_k = int(self.factor * math.log(length))
        weights, delay = torch.topk(corr, top_k, dim=-1)
        # update corr
        tmp_corr = torch.softmax(weights, dim=-1)
        # aggregation
        tmp_values = values.repeat(1, 1, 1, 2)
        delays_agg = torch.zeros_like(values).float()
        for i in range(top_k):
            tmp_delay = init_index + delay[..., i].unsqueeze(-1)
            pattern = torch.gather(tmp_values, dim=-1, index=tmp_delay)
            delays_agg = delays_agg + pattern * (tmp_corr[..., i].unsqueeze(-1))
        return delays_agg   # 这个

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape      # B是batchsize， L是数据长度如label_len+pred_len, H是头数，多头， E是512/H
        _, S, _, D = values.shape       # S是seq_len? D是512/H
        if L > S:     # 由于解码器的第二个自相关模块输入的q和kv的长度不一样，所以需要填充到相同的长度（填零）
            zeros = torch.zeros_like(queries[:, :(L - S), :]).float()
            values = torch.cat([values, zeros], dim=1)
            keys = torch.cat([keys, zeros], dim=1)
        else:
            values = values[:, :L, :, :]
            keys = keys[:, :L, :, :]

        # period-based dependencies
        q_fft = torch.fft.rfft(queries.permute(0, 2, 3, 1).contiguous(), dim=-1)
        k_fft = torch.fft.rfft(keys.permute(0, 2, 3, 1).contiguous(), dim=-1)
        # 上面两个是傅里叶变换之后的向量，维度为（32，8，64，49）
        res = q_fft * torch.conj(k_fft)   # 是将这两个fft得到的向量进行共轭相乘，结果是res，是（32，8，64，49）维度的数据
        corr = torch.fft.irfft(res, dim=-1)    # corr是将这个结果进行逆向傅里叶，得到的是（32，8，64，96）的维度的数据。

        # time delay agg
        if self.training:        # 在训练的时候对values进行处理，主要用到了time_delay_agg_training
            V = self.time_delay_agg_training(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)
            # 应该是利用V和corr输入time_delay_agg_training得到最终的V
        else:
            V = self.time_delay_agg_inference(values.permute(0, 2, 3, 1).contiguous(), corr).permute(0, 3, 1, 2)

        # 最后返回的是V和corr，所以整个步骤的意义主要是对V和利用k和q得到corr
        # 最后返回的是V，也就相当于attention最后的结果？
        if self.output_attention:
            return (V.contiguous(), corr.permute(0, 3, 1, 2))
        else:
            return (V.contiguous(), None)


class AutoCorrelationLayer(nn.Module):
    def __init__(self, correlation, d_model, n_heads, d_keys=None,
                 d_values=None):
        super(AutoCorrelationLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_correlation = correlation
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape    # 调用自相关模块
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_correlation(
            queries,
            keys,
            values,
            attn_mask
        )
        out = out.view(B, L, -1)
            # 这个out也就是V
        return self.out_projection(out), attn
