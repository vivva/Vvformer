import torch
import torch.nn as nn
import torch.nn.functional as F
from Model.Embed import DataEmbedding, DataEmbedding_wo_pos
from Model.AutoCorrelation import AutoCorrelation, AutoCorrelationLayer
from Model.Autoformer_EncDec import Encoder, Decoder, EncoderLayer, DecoderLayer, my_Layernorm, series_decomp
import math
import numpy as np


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.label_len = configs.label_len
        self.pred_len = configs.pred_len

        # 应该就是构建网络结构
        # 1、首先是decmp,调用移动平均来消除趋势项
        kernal_size = configs.moving_avg
        self.decomp = series_decomp(kernal_size)    # 这里的移动平均好像类似于卷积操作？

        # 2、embedding
        self.enc_embedding = DataEmbedding_wo_pos(configs.enc_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        self.dec_embedding = DataEmbedding_wo_pos(configs.dec_in, configs.d_model, configs.embed, configs.freq,
                                                  configs.dropout)
        # 3、encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AutoCorrelationLayer(  # 自相关层
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=configs.output_attention),
                        configs.d_model, configs.n_heads),  # 前面一部分是自相关层，下面的部分是encoder_layer
                    configs.d_model,
                    configs.d_ff,
                    moving_avg=configs.moving_avg,
                    dropout=configs.dropout,
                    activation=configs.activation
                ) for l in range(configs.e_layers)
            ],
            norm_layer=my_Layernorm(configs.d_model)  # 最后加一个norm_layer类
        )
        # 最后返回的是x和attn？但是x是什么呢？


        # 4、decoder
        # Decoder
        self.decoder = Decoder(
            [
                DecoderLayer(
                    AutoCorrelationLayer(    # 堆叠两个自相关层？
                        AutoCorrelation(True, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),
                        configs.d_model, configs.n_heads),
                    AutoCorrelationLayer(
                        AutoCorrelation(False, configs.factor, attention_dropout=configs.dropout,
                                        output_attention=False),   # 前面一部分是自相关层，下面的部分是decoder_layer
                        configs.d_model, configs.n_heads),
                    configs.d_model,   # 模型维度是512
                    configs.c_out,    # output 的size
                    configs.d_ff,   # 全连接层的维度
                    moving_avg=configs.moving_avg,    # 移动平均去除趋势项嵌套在了encoder_layer中
                    dropout=configs.dropout,
                    activation=configs.activation,   # 激活函数
                )
                for l in range(configs.d_layers)    # encoder的层数
            ],
            norm_layer=my_Layernorm(configs.d_model),     # 归一化
            projection=nn.Linear(configs.d_model, configs.c_out, bias=True)    # 最后是一个全连接层
        )   # decoder返回的是x和trend



    def forward_vivva(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # 首先构建dec的输入
        # unsqueeze是在第一维度上增加一个维度
        print('model_forward_x_enc.shape')
        print(x_enc.shape)
        # mean和zeros再这里的shape是[32, 96, 特征维度]，x_enc也是同样的shape
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)    # 趋势项，首先获得趋势项，还要输入到decoder  # repeat是将矩阵A按照给定的din将每个元素重复pred_len次数
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)     #季节项

        seasonal_init, trend_init = self.decomp(x_enc)   # 这里是调用序列分解函数，得到残差项和趋势项，shape都是【32，96，7】

        # 将非预测部分的季节or趋势和预测部分的mean or zero进行拼接

        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)      # 同理，[32,144,7]
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros],
                                  dim=1)  # 第二维度加上label_len=48，shape变为[32,144,7]

        # 先编码，输入到enc
        # enc_embedding是将x_enc(Conv1D)和x_enc_mark(Linear)的shape编码都为【32，96，512】，然后相加，得到编码器的输入 enc_out：[32, 96, 512]
        enc_out = self.enc_embedding(x_enc, x_mark_enc)     # shape变为32，96，512
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)    # encoder输出的是季节项和attn

        # decoder也是同样先编码再输入到decoder,并将初始的趋势项h和encoder的输出都输入其中
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        dec_out = trend_part + seasonal_part

        # 不返回attention

        return dec_out[:, -self.pred_len:, :]    # [B, L, D]   # BLD应该是batchsize、预测长度和数据维度？

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec,
                enc_self_mask=None, dec_self_mask=None, dec_enc_mask=None):
        # decomp init
        # 这应该是decoder的输入,decoder解码器输入的有:1/zero的季节初始化    2/input的均值(x_enc)的趋势项
        mean = torch.mean(x_enc, dim=1).unsqueeze(1).repeat(1, self.pred_len, 1)      # 这个应该是趋势项
        zeros = torch.zeros([x_dec.shape[0], self.pred_len, x_dec.shape[2]], device=x_enc.device)    # zero的是季节项

        # 将x_enc输入到decomp得到季节项和趋势项
        seasonal_init, trend_init = self.decomp(x_enc)

        # decoder input,输入的是用原来的(label部分)+mean和zero填充的要预测的部分
        trend_init = torch.cat([trend_init[:, -self.label_len:, :], mean], dim=1)
        seasonal_init = torch.cat([seasonal_init[:, -self.label_len:, :], zeros], dim=1)
        # enc,encoder在词编码之后输入到encoder中得到输出
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=enc_self_mask)
        # dec,将季节项和时间戳进行编码,将编码后的和encoder输出的/原始的趋势项进行输入,得到最终的decoder输出的趋势项和季节项
        dec_out = self.dec_embedding(seasonal_init, x_mark_dec)
        seasonal_part, trend_part = self.decoder(dec_out, enc_out, x_mask=dec_self_mask, cross_mask=dec_enc_mask,
                                                 trend=trend_init)
        # final,最后将趋势项和季节项合并得到最终的输出结果
        dec_out = trend_part + seasonal_part

        # 看是否输出attention
        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]


