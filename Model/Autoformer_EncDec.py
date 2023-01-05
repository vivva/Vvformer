import torch
import torch.nn as nn
import torch.nn.functional as F


class my_Layernorm(nn.Module):
    """
    季节性部分的特殊设计层
    """
    def __init__(self, channels):    # channel是模型维度为512
        super(my_Layernorm, self).__init__()
        self.layernorm = nn.LayerNorm(channels)

    def forward(self, x):
        x_hat = self.layernorm(x)
        bias = torch.mean(x_hat, dim=1).unsqueeze(1).repeat(1, x.shape[1], 1)
        return x_hat - bias


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    移动平均块以突出时间序列的趋势，突出
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        # 时间序列两端的填充
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)     # [32, 1,特征维度].repeat是将第二个维度扩充到12，特征维度为7或20
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        # 这个end是将倒数第一个维度进行扩充变成了12
        # 所以到了这里，front和end的维度都是变成了（32，12，7）。而原来的输入的x的维度是（32，96，7）
        # 为了形成对比方便理解记录一下。一前一后都是取这个原来的x的其中一个维度进行复制12份，而原来的x的维度是（32，96，7）

        x = torch.cat([front, x, end], dim=1)   # 三个矩阵进行叠加，形成了一个维度为（32，120，7）的新的向量
        x = self.avg(x.permute(0, 2, 1))  # 对这个x进行卷积的操作，permute是将tensor的维度进行换位。这一步结束后，x的维度是（32，7，96）应该是32，7，120吧？
        # 换位置是因为avgpool是再最后一维操作，而需要pooling的是序列的那个维度，所以换到最后。即96长度的那个数据
        x = x.permute(0, 2, 1)     # x = x.permute(0, 2, 1)。这一步是对x进行维度的换，经过这一次x的维度变成了（32，96，7），应该是32，96，120？？？
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    序列分解模块
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        # 这里的x就是刚开始的x_enc
        moving_mean = self.moving_avg(x)
        res = x - moving_mean      # 应该是用原始的减去移动平均得到的趋势项，得到剩下的季节项
        return res, moving_mean


class EncoderLayer(nn.Module):
    """
    Autoformer encoder layer with the progressive decomposition architecture
    """
    def __init__(self, attention, d_model, d_ff=None, moving_avg=25, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model    # 都是2048
        self.attention = attention    # 这里的attention是自相关
        # 中间的conv1d进行信息提取
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        # ------1、自相关
        # 相对于transformer的attention来说，多一个   inner_correlation部分

        new_x, attn = self.attention(    # attention返回的是V，new_x应该就是V？
            x, x, x,
            attn_mask=attn_mask
        )
        x = x + self.dropout(new_x)

        # ----------2、时序分解
        x, _ = self.decomp1(x)    # 2、序列分解，res和moving_avg，将得到的季节项也就是x幅值给y
        y = x

        # ---------3、feed forward
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))     # 对季节项进行conv1d特征提取并dropout
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        # ----------4、时序分解
        # 对卷积后的季节项进一步时序分解，得到更深层的季节项
        res, _ = self.decomp2(x + y)    # 再将季节项和经过卷积、激活、dropout之后的季节项相加进行序列分解得到新的季节项和趋势项
        return res, attn


class Encoder(nn.Module):
    """
    Autoformer encoder
    主要有attention层、卷积层和归一化层
    """
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)    # 自相关机制，这个也就是encoder layer？
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        if self.conv_layers is not None:
            for attn_layer, conv_layer in zip(self.attn_layers, self.conv_layers):
                x, attn = attn_layer(x, attn_mask=attn_mask)   # atten好像返回的是什么延时聚合？V？应该是自相关的东西，这部分还不太懂
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns


class DecoderLayer(nn.Module):
    """
    Autoformer decoder layer with the progressive decomposition architecture
    """
    def __init__(self, self_attention, cross_attention, d_model, c_out, d_ff=None,
                 moving_avg=25, dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention     # 这里的应该是自相关
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1, bias=False)
        self.decomp1 = series_decomp(moving_avg)
        self.decomp2 = series_decomp(moving_avg)
        self.decomp3 = series_decomp(moving_avg)
        self.dropout = nn.Dropout(dropout)
        self.projection = nn.Conv1d(in_channels=d_model, out_channels=c_out, kernel_size=3, stride=1, padding=1,
                                    padding_mode='circular', bias=False)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(
            x, x, x,
            attn_mask=x_mask
        )[0])  # 这个得到的x应该是？注意力之类的东西？
        x, trend1 = self.decomp1(x)    # 根据注意力结果？进行序列分解之后，得到趋势项和季节项？
        x = x + self.dropout(self.cross_attention(
            x, cross, cross,
            attn_mask=cross_mask
        )[0])    # cross attention？
        x, trend2 = self.decomp2(x)
        y = x
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        x, trend3 = self.decomp3(x + y)

        residual_trend = trend1 + trend2 + trend3     # 三个趋势项相加？将每次趋势分解出来的结果进行相加
        residual_trend = self.projection(residual_trend.permute(0, 2, 1)).transpose(1, 2)     # 再将这个进行？linear？
        return x, residual_trend


class Decoder(nn.Module):   # 输入的是  DecoderLayer，d——layer，norm_layer, Linear
    """
    Autoformer encoder
    """
    def __init__(self, layers, norm_layer=None, projection=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)    # ModuleList应该是decoderLater里定义的网络结构
        self.norm = norm_layer
        self.projection = projection

    def forward(self, x, cross, x_mask=None, cross_mask=None, trend=None):
        for layer in self.layers:
            x, residual_trend = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
            trend = trend + residual_trend    # 季节项+ 趋势项

        if self.norm is not None:
            x = self.norm(x)

        if self.projection is not None:
            x = self.projection(x)
        return x, trend
