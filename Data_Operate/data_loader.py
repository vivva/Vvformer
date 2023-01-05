import os

import pandas as pd
from datetime import datetime

from sklearn.preprocessing import StandardScaler
from torch.utils.data import Dataset
from utils.timefeatures import time_features



class Dataset_ETT_hour(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        # size一般为[seq_len, label_len, pred_len]
        # 这里应该是定义一下每个样本以及需要预测的起点与长度？
        if size == None:
            self.seq_len = 24 * 4 * 4   # 384，可能是一个样本是384，预测长度为96
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq

        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))

        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]    #
        border2 = border2s[self.set_type]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:   # 1
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp

    def __read_data_vivva__(self):
        # 主要还是读取数据，包括特征数据以及时间戳；划分训练测试的数据边界，以及一些时间戳的编码？好像是
        self.scaler = StandardScaler()    # 归一化，保证每个维度方差为1，均值为0，加快梯度下降求最优解的速度
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))   # df_raw最先获取的是ETTh1    shape是（17420，8）
        # 边界的划分有啥依据？
        border1s = [0, 12 * 30 * 24 - self.seq_len, 12 * 30 * 24 + 4 * 30 * 24 - self.seq_len]
        border2s = [12 * 30 * 24, 12 * 30 * 24 + 4 * 30 * 24, 12 * 30 * 24 + 8 * 30 * 24]
        border1 = border1s[self.set_type]            # 看是训练还是test or val[0,8544,11424]
        border2 = border2s[self.set_type]        # {'train': 0, 'test': 1, 'pred': 2}[0,8544,11424]

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]    # 这是特征个数，除了第一列的都是特征
            df_data = df_raw[cols_data]    # 根据列名提取特征列？
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:    # 如果归一化的话，默认是归一化的
            train_data = df_data[border1s[0]:border2s[0]]    # 从0到一年，以小时为单位
            self.scaler.fit(train_data.values)    # 这个是计算每一列的均值和方差
            data = self.scaler.transform(df_data.values)    # 直接将矩阵标准化，用到的是上一步计算出来的均值和方差
        # 机器学习中，对于训练数据，用fit_transformer进行标准化，测试数据用transformer
        # fit是计算矩阵的每一列平均值和方差
        # transformer是根据均值和方差，将矩阵标准化
        # fit_transformer是求均值和方差并标准化矩阵
        else:
            data = df_data.values

        # 时间戳
        df_stamp = df_raw[['date']][border1, border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)

        # 时间戳进行编码，编码后的shape为【8640，4】，因为是hour，所以是4
        if self.timeenc == 0:   # 默认为0
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)    # 这个语法不是很明白什么意思
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:     # 1是不对时间戳进行时间编码？
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)    # 没太理解这个函数的作用
            data_stamp = data_stamp.transpose(1, 0)

        # 为什么x和y是一样的，data是特征数据
        self.data_x = data[border1, border2]    # x和y都是【8640，7】
        self.data_y = data[border1, border2]
        self.data_stamp = data_stamp     # 这个是【8640，4】

    def __getitem__(self, index):
        s_begin = index    # index应该就是不断的滑窗的序号
        s_end = s_begin + self.pred_len     # s_begin 到s_end只有一个预测长度
        r_begin = s_end - self.label_len       # 算是起始令牌长度
        r_end = r_begin + self.label_len + self.pred_len    # 也只有一个预测长度

        # 上面的长度是怎么划分的还不知道
        # x和y一个是输入到encoder，一个输入到decoder？
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark


    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transformer(self, data):
        # 将标准化后的数据转换为原始数据
        return self.scaler.inverse_transform(data)   # 估计是直接用的上面fit计算出来的mean和std


class Dataset_Custom(Dataset):
    def __init__(self, root_path, flag='train', size=None,
                 features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, timeenc=0, freq='h'):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]

            # init
            assert flag in ['train', 'test', 'val']
            type_map = {'train': 0, 'val': 1, 'test': 2}
            self.set_type = type_map[flag]

            self.features = features
            self.target = target
            self.scale = scale
            self.timeenc = timeenc
            self.freq = freq

            self.root_path = root_path
            self.data_path = data_path
            self.__read_data__()



    def __read_data__(self):
        # 读取数据
        self.scaler = StandardScaler()     # 使经过处理的数据符合标准正态分布，即均值为0，标准差为1；减均值，然后除以标准差
        df_raw = pd.read_csv(os.path.join(self.root_path,
                                          self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        cols = list(df_raw.columns)
        cols.remove(self.target)
        cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]   # 不管如何，构建一个具有时间列和target列的

        # 划分训练集、测试集
        num_train = int(len(df_raw) * 0.7)  # 70%作为训练集
        num_test = int(len(df_raw) * 0.2)
        num_vali = len(df_raw) - num_train - num_test

        # 训练集、测试集和验证集彼此重复seq_len
        border1s = [0, num_train - self.seq_len, len(df_raw) - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, len(df_raw)]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]   # CSDN上说：？这里验证集会有泄漏，可能导致训练loss和验证loss很低，但是 test loss 却很高

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        df_stamp = df_raw[['date']][border1:border2]
        df_stamp['date'] = pd.to_datetime(df_stamp.date)
        if self.timeenc == 0:    #  # 如果 args.embed != 'timeF'，就会把时间编码为 month，day，weekday，hour 四个数
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)     # +1 =星期几
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:     # args.embed == 'timeF'
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)     # 根据传入的 freq 对时间戳进行解析
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]
        self.data_stamp = data_stamp



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1    # ？ 这样写估计是最后一个能去到完整seq_len+ pred_len序列

    def inverse_transformer(self, data):        # 数据恢复到原来尺度
        return self.scaler.inverse_transform(data)




class Dataset_Pred():
    def __init__(self, root_path, flag='pred', size=None, features='S', data_path='ETTh1.csv',
                 target='OT', scale=True, inverse=False, timeenc=0, freq='15min', cols=None):
        if size == None:
            self.seq_len = 24 * 4 * 4
            self.label_len = 24 * 4
            self.pred_len = 24 * 4
        else:
            self.seq_len = size[0]
            self.label_len = size[1]
            self.pred_len = size[2]
        # init
        assert flag in ['pred']

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()


    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        '''
        df_raw.columns: ['date', ...(other features), target feature]
        '''
        if self.cols:
            cols = self.cols.copy()
            cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove(self.target)
            cols.remove('date')
        df_raw = df_raw[['date'] + cols + [self.target]]
        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features == 'M' or self.features == 'MS':
            cols_data = df_raw.columns[1:]
            df_data = df_raw[cols_data]
        elif self.features == 'S':
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        tmp_stamp = df_raw[['date']][border1:border2]   # 长度为aeq_len
        tmp_stamp['date'] = pd.to_datetime(tmp_stamp.date)    # pd.to_datetime解析时间
        pred_dates = pd.date_range(tmp_stamp.date.values[-1], periods=self.pred_len + 1, freq=self.freq)   # 这个是得到需要预测的时间段

        df_stamp = pd.DataFrame(columns=['date'])
        df_stamp.date = list(tmp_stamp.date.values) + list(pred_dates[1:])    # 这个是需要输出的所有的时间段
        if self.timeenc == 0:
            df_stamp['month'] = df_stamp.date.apply(lambda row: row.month, 1)
            df_stamp['day'] = df_stamp.date.apply(lambda row: row.day, 1)
            df_stamp['weekday'] = df_stamp.date.apply(lambda row: row.weekday(), 1)
            df_stamp['hour'] = df_stamp.date.apply(lambda row: row.hour, 1)
            df_stamp['minute'] = df_stamp.date.apply(lambda row: row.minute, 1)
            df_stamp['minute'] = df_stamp.minute.map(lambda x: x // 15)
            data_stamp = df_stamp.drop(['date'], 1).values
        elif self.timeenc == 1:
            data_stamp = time_features(pd.to_datetime(df_stamp['date'].values), freq=self.freq)
            data_stamp = data_stamp.transpose(1, 0)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp



    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]   # data_x是归一化之后的数据
        if self.inverse:
            seq_y = self.data_x[r_begin:r_begin + self.label_len]
        else:
            seq_y = self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.fit_transform(data)












def Data_Format_Transfor():
    # 转换时间格式，将时间格式转换为标准的时间形式，数据的第一列是date
    df = pd.read_csv("a.csv")
    df['data'] = pd.to_datetime(df['data'], format='%Y/%m/%d %H:%M')

    df['data'] = df['data'].dt.strftime('%Y-%m-%d %H:%M:%S')
    print(df)
    df = df.to_csv('b.csv', index=False)
    print(df)











