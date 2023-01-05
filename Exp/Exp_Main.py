import os
from utils.tools import EarlyStopping, adjust_learning_rate, visual
from Data_Operate.data_factory import data_provider
from Exp.Exp_basic import Exp_Basic
import time
from torch import optim
import torch.nn as nn
import torch
import numpy as np
from Model import model as Autoformer
from utils.metrics import metric



class exp_main(Exp_Basic):
    def __init__(self, args):   # 继承basic
        super(exp_main, self).__init__(args)

    def _build_model(self):

        model_dict = {
            'Autoformer': Autoformer,
        }

        # model = model_dict[self.args.model]
        model = model_dict['Autoformer'].Model(self.args).float()

        # 是否使用GPU
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_id)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)    # 调用的是data_factory 中的data_provider
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion


    # 在这里定义一个物理损失函数,在train'、中的损失函数将其加入
    def _Phy_loss(self, model, AX1, AX2):
        tolerance = 0
        Aout1 = model.predict(AX1)
        Aout2 = model.predict(AX2)
        # 接下来应该是按照物理机理模型进行带入？或者直接定义高低点来进行损失函数?
        udendiff = (density(uout1) - density(uout2))  # 计算相应的密度和差？
        percentage_phy_incon = np.sum(udendiff > tolerance) / udendiff.shape[0]  # 计算%
        phy_loss = np.mean(relu(udendiff))  # 密度差小于0就是0？大于0就惩罚？
        return phy_loss, percentage_phy_incon




    def train(self, setting):
        print('hello')
        # 读取数据
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()    # 设定时间？

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # 定义损失函数和优化器
        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        # 训练模型
        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()    # 不用定义网络结构，直接调用定义好的网络结构
            epoch_time = time.time()
            # 将读取出的数据按照batch输入进模型中
            # batch_x: batch_size, seq_len, d_in
            # batch_x_mark: batch_size, seq_len, 4
            # batch_x是只有原来的数据，即seq_len,是输入到encoder里面的
            # batch_y是从起始令牌长度（部分原始数据）加要预测的数据，是要输入到decoder里面的
            # batch_y: batch_size, label_len+pred_len, d_in
            # batch_y_mark: batch_size, label_len+pred_len, 4
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                # 首先将模型的梯度清零
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 输入到模型中，包括decoder和encoder
                # 输入解码器decoder的有标记为0的季节项和均值的趋势项
                # 这个是解码器的输入
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()    # 解码器
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)      # 逗号中间分别是几个维度，冒号代表所有维度，但是一个：一个数字是什么意思呢
                # [batchsize, 96, 21]对batch的切片操作应该是【batch,从最后到预测长度部分，21】【batch, 从开始头部到起始预测处，21】
                print('dec_inp')
                # 单纯只输出outputs，不输出attention
                if self.args.output_attention:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark, batch_y)
                print('output')
                # outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                # outputs
                # 如果是多变量预测单变量，f_dim=-1
                # f_dim = -1 if self.args.feature == 'MS' else f_dim = 0
                f_dim = -1 if self.args.features == 'MS' else 0
                # 只提取出预测长度部分的模型输出
                outputs = outputs[:, -self.args.pred_len:, f_dim:]      # 预测数据
                print('outputs1')
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)  # 真实数据

                # 计算损失
                loss = criterion(outputs, batch_y)
                # 这个应该是将每个batch的loss都添加到这里
                train_loss.append(loss.item())
                    # 如果已经循环了100次的batch，那么即为一个epoch
                    # i是遍历train_loader相当于train_loader的长度？i相当于是train_loader的数据下标
                if (i+1) % 100 ==0:   # 返回除法的余数
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 默认不使用混合精度，所以直接损失函数 回传并更新梯度
                loss.backward()
                model_optim.step()

            # 跳出for i的循环，即完成一次epoch的训练
            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            # 在train里面直接调用vali和test函数
            vali_loss =self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            print("Epoch:{0}, Step:{1}, | Train Loss{2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                    epoch+1, train_steps, train_loss, vali_loss, test_loss))

            # 提前停止并保存模型
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            # 每过一个epoch？就调整学习率
            adjust_learning_rate(model_optim, epoch+1, self.args)

        # 在所有epoch训练完之后，保存最好的模型
        best_model_peth = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_peth))

        return self.model


    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        self.model.eval()
        # 验证的时候不需要更新梯度
         # 相对于train，没有损失回传，没有梯度更新，只是计算一下loss
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)


                # 开始一个批次的验证
                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)
                total_loss.append(loss)

            total_loss = np.average(total_loss)
            self.model.train()    # 为什么这里又将模式调回训练模型？
            return total_loss




    def test(self, setting, test=0):
        # 读取数据
        test_data, test_loader = self._get_data(flag='test')
        # 如果要是test的话，需要加载模型
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()   # 开启模型的测试模型
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()     # 将要预测的部分变为0，
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)    # 连接预测起点前的数据和变为0的数据

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]

                f_dim = -1 if self.args.feature == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                pred = outputs
                true = batch_y
                preds.append(pred)
                trues.append(true)
                if i % 20 == 0:    # i循环20次？
                    input = batch_x.detach().cpu().numpy()
                    gt = np.concatenate((input[0, :, -1], true[0, :, -1]), axis=0)   # 为什么要是0？
                    pd = np.concatenate((input[0, :, -1], pred[0, :, -1]), axis=0)
                    visual(gt, pd, os.path.join(folder_path, str(i) + '.pdf'))

                preds = np.array(preds)
                trues = np.array(trues)
                print('test shape:', preds.shape, trues.shape)
                preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
                trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
                print('test shape:', preds.shape, trues.shape)

                # result save
                folder_path = './results/' + setting + '/'
                if not os.path.exists(folder_path):
                    os.makedirs(folder_path)

                mae, mse, rmse, mape, mspe = metric(preds, trues)
                print('mse:{}, mae:{}'.format(mse, mae))
                f = open("result.txt", 'a')
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}'.format(mse, mae))
                f.write('\n')
                f.write('\n')
                f.close()

                np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
                np.save(folder_path + 'pred.npy', preds)
                np.save(folder_path + 'true.npy', trues)

                return



    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')

        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path + '/' + 'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        preds = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[2]]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                pred = outputs.detach().cpu().numpy()  # .squeeze()
                preds.append(pred)

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])  # reshape？

        # result save
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        np.save(folder_path + 'real_prediction.npy', preds)

        return


