# CIFAR_corr_long_offline_RL.py
# 在原 CIFAR_corr_long_offline_new2.py 基础上加入连续动作空间的策略梯度（REINFORCE）agent，
# 用以动态调整 correction_rate ∈ [0,1].

import queue
import threading
import argparse

import numpy.random
from matplotlib import pyplot as plt
import torch
from torch import nn
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import math
import os.path
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
from pandas import DataFrame
import random
import numpy as np
import os
from torchvision import datasets, models
import time
import multiprocessing
import copy
from scipy.stats import dirichlet

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# To print in color -------test/train of the client side
def prRed(skk):
    print("\033[91m {}\033[00m".format(skk))


def prGreen(skk):
    print("\033[92m {}\033[00m".format(skk))


# =====================================================================================================
#                           Client-side Model definition
# =====================================================================================================
# Model at client side
class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * self.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes * self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * self.expansion)
            )

    def forward(self, x):
        LeakyReLu = nn.LeakyReLU()
        out = LeakyReLu(self.bn1(self.conv1(x)))
        out = LeakyReLu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = LeakyReLu(out)
        return out


class ResNet50_client_side(nn.Module):
    def __init__(self):
        super(ResNet50_client_side, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        LeakyReLu = nn.LeakyReLU()
        out = LeakyReLu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        return out


# =====================================================================================================
#                           Server-side Model definition
# =====================================================================================================
# Model at server side
class ResNet50_server_side(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_server_side, self).__init__()
        self.in_planes = 256  # 由于已经经过了self.layer1，所以更新in_planes

        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.layer2(x)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


# ====================================================================================================
#                                  Server Side Programs
# ====================================================================================================
# Federated averaging: FedAvg
def FedAvg(w, corrections, model_type, weights=None):
    """
    Federated averaging with optional sample-count weights.

    Args:
        w: list of state_dicts (each a dict of tensors)
        corrections: unused here but kept for compatibility with callers
        model_type: 'client' or 'server' to pick param keys
        weights: optional list of non-negative numbers with same length as w
    Returns:
        w_avg: aggregated state_dict
    """
    if not w:
        return {}

    # ensure tensors are float
    w = [{k: v.float() for k, v in params.items()} for params in w]

    if model_type == 'client':
        w_avg = copy.deepcopy(w[0])
        param_keys = net_glob_client.state_dict().keys()
    elif model_type == 'server':
        w_avg = copy.deepcopy(w[0])
        param_keys = net_glob_server.state_dict().keys()
    else:
        raise ValueError("Invalid model_type")

    # single contributor -> return copy
    if len(w) == 1:
        return w_avg

    # prepare weights
    use_weights = False
    if weights is not None:
        try:
            if len(weights) == len(w):
                w_tensor = torch.tensor(weights, dtype=torch.float)
                s = float(w_tensor.sum())
                if s > 0:
                    w_norm = (w_tensor / s).tolist()
                    use_weights = True
                else:
                    use_weights = False
            else:
                print(f"[Warning] weights length ({len(weights)}) != number of models ({len(w)}), falling back to uniform average")
                use_weights = False
        except Exception:
            use_weights = False

    # aggregate
    for k in param_keys:
        if use_weights:
            total = torch.zeros_like(w_avg[k])
            for i, params in enumerate(w):
                total = total + params[k].cpu() * float(w_norm[i])
            w_avg[k] = total
        else:
            # uniform average
            total = w_avg[k].clone()
            for params in w[1:]:
                total += params[k].cpu()
            w_avg[k] = total / len(w)

    return w_avg


def calculate_accuracy(fx, y):
    preds = fx.max(1, keepdim=True)[1]
    correct = preds.eq(y.view_as(preds)).sum()
    acc = 100.00 * correct.float() / preds.shape[0]
    return acc


# Server-side function associated with Training
def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, net_glob_server, lr, criterion,
                 batch_acc_train, batch_loss_train, count1, loss_train_collect_user, acc_train_collect_user,
                 idx_collect, idx_disconnected, idx_round_disconnected,
                 num_users):
    global l_epoch_check, fed_check
    net_glob_server = net_glob_server.to('cuda:0')  # 将模型移到 GPU 上
    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)

    # train and update
    optimizer_server.zero_grad()

    fx_client = fx_client
    fx_client = fx_client.requires_grad_(True)
    y = y

    # ---------forward prop-------------
    fx_server = net_glob_server(fx_client)

    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------
    loss.backward()
    if torch.isnan(fx_client.grad).any():
        print(f"[Error] Server 梯度包含NaN，Client {idx} 的参数将被标记为无效")
        idx_round_disconnected.append(idx)
        return None, net_glob_server  # 返回None表示梯度无效
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # server-side model net_glob_server is global so it is updated automatically in each pass to this function

    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    if count1 == len_batch * l_epoch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)  # it has accuracy for one batch
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))

        # If one local epoch is completed, after this a new client will come
        if l_epoch_count == l_epoch - 1:

            l_epoch_check = True  # to evaluate_server function - to check local epoch has completed or not

            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            # collect the id of each new user
            if idx not in idx_collect:
                if idx in idx_disconnected:
                    print(f"[Warning] Client{idx} 在 idx_disconnected 中，可能存在竞态条件")
                else:
                    idx_collect.append(idx)

        print(f"[Debug] idx_collect: {idx_collect}")
        print(f"[Debug] idx_disconnected: {idx_disconnected}")

        # This is to check if all users are served for one round --------------------
        if len(idx_collect) + len(idx_round_disconnected) == num_users:
            fed_check = True  # to evaluate_server function  - to check fed check has hitted

            if len(acc_train_collect_user) > 0:
                acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
                loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)
            else:
                if len(acc_train_collect) == 0:
                    acc_avg_all_user_train = 0
                    loss_avg_all_user_train = 0
                else:
                    acc_avg_all_user_train = acc_train_collect[-1]
                    loss_avg_all_user_train = loss_train_collect[-1]

            global acc_avg_all_user_train_global, loss_avg_all_user_train_global
            acc_avg_all_user_train_global = acc_avg_all_user_train
            loss_avg_all_user_train_global = loss_avg_all_user_train
            acc_train_collect.append(acc_avg_all_user_train)
            print('[train_server] acc_train_collect appended once, current length:', len(acc_train_collect))
            print('[train_server] current idx_collect:', idx_collect)
            print('[train_server] current idx_disconnected:', idx_disconnected)
            loss_train_collect.append(loss_avg_all_user_train)

    return dfx_client, net_glob_server


# Server-side functions associated with Testing
def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train_global, loss_avg_all_user_train_global
    net_glob_server = net_glob_server.to('cuda:0')  # 将模型移到 GPU 上
    net_glob_server.eval()

    with torch.no_grad():
        fx_client = fx_client.to('cuda:0')
        y = y.to('cuda:0')

        fx_server = net_glob_server(fx_client)

        # calculate loss
        loss = criterion(fx_server, y)
        # calculate accuracy
        acc = calculate_accuracy(fx_server, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / len(batch_acc_test)
            loss_avg_test = sum(batch_loss_test) / len(batch_loss_test)

            batch_acc_test = []
            batch_loss_test = []
            count2 = 0

            print('Client{} Test =>                   \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, acc_avg_test,
                                                                                           loss_avg_test))

            # if a local epoch is completed
            if l_epoch_check:
                l_epoch_check = False

                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

            # if all users are served for one round ----------
            if fed_check:
                fed_check = False

                acc_avg_all_user = sum(acc_test_collect_user) / len(acc_test_collect_user)
                loss_avg_all_user = sum(loss_test_collect_user) / len(loss_test_collect_user)

                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                print('[evaluate_server] acc_test_collect appended once, current length:', len(acc_test_collect))
                acc_test_collect_user = []
                loss_test_collect_user = []

                print("====================== SERVER V1==========================")
                print(' Train: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell,
                                                                                          acc_avg_all_user_train_global,
                                                                                          loss_avg_all_user_train_global))
                print(' Test: Round {:3d}, Avg Accuracy {:.3f} | Avg Loss {:.3f}'.format(ell, acc_avg_all_user,
                                                                                         loss_avg_all_user))
                print("==========================================================")

    return


# 在全局作用域中定义新的全局变量
acc_avg_all_user_train_global = 0
loss_avg_all_user_train_global = 0


# ==============================================================================================================
#                                       Clients Side Program
# ==============================================================================================================
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


# Client-side functions associated with Training and Testing
class Client(object):
    def __init__(self, net_client_model, idx, lr, net_glob_server, criterion, count1, idx_collect, num_users, running,
                 dataset_train=None, dataset_test=None, idxs=None, idxs_test=None, heartbeat_queue=None, disconnect_prob=0.001, idx_disconnected=None, is_disconnected=False, idx_disconnected_time=None, idx_round_disconnected=None, disconnect_seed=0, disconnect_round = 1, local_ep = 1):
        self.disconnect_prob = disconnect_prob  # 断开概率
        self.is_disconnected = is_disconnected  # 是否断开
        self.heartbeat_queue = heartbeat_queue
        self.idx = idx
        self.lr = lr
        self.local_ep = local_ep
        self.net_glob_server = net_glob_server
        self.criterion = criterion
        self.batch_acc_train = []
        self.batch_loss_train = []
        self.loss_train_collect_user = []
        self.acc_train_collect_user = []
        self.count1 = 0
        self.idx_collect = idx_collect
        self.num_users = num_users
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=1024, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=1024, shuffle=True)
        self.disconnect_seed = disconnect_seed
        self.rng = numpy.random.default_rng(seed=self.disconnect_seed + self.idx)
        self.disconnect_round = disconnect_round

        self.idx_disconnected = idx_disconnected
        self.idx_round_disconnected = idx_round_disconnected
        self.idx_disconnected_time = idx_disconnected_time
        self.running = running
        # 新增心跳管理
        self.status = "idle"  # idle, training, testing
        self.heartbeat_interval = 5 # 心跳间隔
        self.stop_heartbeat_flag = False
        # 心跳线程在初始化最后启动（确保属性已创建）
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def send_heartbeat(self):
        try:
            # while self.running.value and not self.stop_heartbeat_flag:
            while not self.stop_heartbeat_flag:
                if not self.is_disconnected:
                    # 仅在未断开时检查是否断开
                    random_num = self.rng.random()
                    # 补充 seed
                    self.rng = numpy.random.default_rng(seed=self.disconnect_seed + self.idx)
                    # for debugging print random number
                    print(f"[send_heartbeat] Client{self.idx} 随机数: {random_num}")
                    self.is_disconnected = random_num < self.disconnect_prob and self.status == "training"
                    if self.is_disconnected:
                        # for debugging print random number
                        print(f"[Random] Client{self.idx} 随机数: {random_num}")
                        print(f"[Disconnect] Client{self.idx} 断开 (概率{self.disconnect_prob * 100}%)")
                        # 发送断开信号后休眠
                        self.heartbeat_queue.put((self.idx, "disconnected", time.strftime("%Y-%m-%d %H:%M:%S")))

                        # 在idx_disconnected中记录已断开的客户端
                        if self.idx not in self.idx_disconnected:
                            # 添加检查避免竞态条件
                            if self.idx in self.idx_collect:
                                print(f"[Warning] Client{self.idx} 在idx_collect中，可能存在竞态条件")
                                print(f"[send_heartbeat] 触发保护机制，不将 {self.idx} 添加到 idx_disconnected")
                                continue
                            else:
                                idx_disconnected.append(self.idx)
                                idx_round_disconnected.append(self.idx)

                        idx_disconnected_time[self.idx] = self.disconnect_round
                        time.sleep(self.heartbeat_interval)
                        continue

                if self.is_disconnected:
                    # 断开后持续休眠，不再尝试连接
                    time.sleep(self.heartbeat_interval)
                    continue

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.heartbeat_queue.put((self.idx, self.status, timestamp))
                time.sleep(self.heartbeat_interval)
        except (EOFError, BrokenPipeError):
            print(f"[Info] Client{self.idx} 心跳线程因连接断开而退出")

        # 线程结束，打印退出信息
        print(f"[Info] Client{self.idx} 心跳线程结束")

    def stop_heartbeat(self):
        """停止心跳线程"""
        self.stop_heartbeat_flag = True
        if self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join()  # 等待线程结束

    def update_fed_check(self):
        """新增鲁棒性保证 如果最后一个client在训练时退出将导致fed_check无法置为True 在这里再做一次检查"""
        global l_epoch_check, fed_check
        if len(self.idx_collect) + len(self.idx_round_disconnected) == self.num_users:
            fed_check = True
            # 确保列表中有数据再计算平均
            if len(self.acc_train_collect_user) > 0:
                acc_avg_all_user_train = sum(self.acc_train_collect_user) / len(self.acc_train_collect_user)
                loss_avg_all_user_train = sum(self.loss_train_collect_user) / len(self.loss_train_collect_user)
            else:
                if len(acc_train_collect) == 0:
                    acc_avg_all_user_train = 0
                    loss_avg_all_user_train = 0
                else:
                    acc_avg_all_user_train = acc_train_collect[-1]
                    loss_avg_all_user_train = loss_train_collect[-1]

            global acc_avg_all_user_train_global, loss_avg_all_user_train_global
            acc_avg_all_user_train_global = acc_avg_all_user_train
            loss_avg_all_user_train_global = loss_avg_all_user_train
            acc_train_collect.append(acc_avg_all_user_train)
            loss_train_collect.append(loss_avg_all_user_train)
            if len(acc_test_collect) == 0:
                acc_test_collect.append(0)
                loss_test_collect.append(0)
            else:
                acc_test_collect.append(acc_test_collect[-1])
                loss_test_collect.append(loss_test_collect[-1])
            print('[update_fed_check] acc_test_collect appended once, current length:', len(acc_test_collect))
        return None, None

    def train(self, net):
        global l_epoch_check, fed_check
        if self.is_disconnected:
            print(f"[Skip] Client{self.idx} 断开，跳过训练")
            if self.idx not in self.idx_disconnected:
                if self.idx in self.idx_collect:
                    print(f"[Warning] Client{self.idx} 在 idx_collect 中，可能存在竞态条件")
                else:
                    idx_disconnected.append(self.idx)
            if fed_check == False and len(self.idx_collect) + len(self.idx_disconnected) == self.num_users:
                self.update_fed_check()
            return None, None

        else:
            try:
                self.status = "training"  # 更新状态
                net = net.to('cuda:0')  # 显式移到 GPU
                self.net_glob_server = self.net_glob_server.to('cuda:0')
                net.train()
                optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)

                for iter in range(self.local_ep):

                    # 检查是否断开连接
                    if self.is_disconnected:
                        print(f"[Abort] Client{self.idx} 训练期间断开，终止训练")
                        if self.idx not in self.idx_disconnected:
                            if self.idx in self.idx_collect:
                                print(f"[Warning] Client{self.idx} 在 idx_collect 中，可能存在竞态条件")
                            else:
                                idx_disconnected.append(self.idx)
                        if fed_check == False and len(self.idx_collect) + len(self.idx_disconnected) == self.num_users:
                            self.update_fed_check()
                        break
                    len_batch = len(self.ldr_train)

                    for batch_idx, (images, labels) in enumerate(self.ldr_train):

                        # 检查是否断开连接
                        if self.is_disconnected:
                            print(f"[Abort] Client{self.idx} 训练期间断开，终止训练")
                            if self.idx not in self.idx_disconnected:
                                if self.idx in self.idx_collect:
                                    print(f"[Warning] Client{self.idx} 在 idx_collect 中，可能存在竞态条件")
                                else:
                                    idx_disconnected.append(self.idx)
                            if fed_check == False and len(self.idx_collect) + len(
                                    self.idx_disconnected) == self.num_users:
                                self.update_fed_check()
                            break

                        images, labels = images.to('cuda:0'), labels.to('cuda:0')

                        optimizer_client.zero_grad()
                        # ---------forward prop-------------
                        fx = net(images)
                        client_fx = fx.clone().detach()

                        # transmit client_fx to server
                        client_fx = client_fx.to('cuda:0')

                        self.count1 = self.count1 + 1

                        print('client ', self.idx, ' :', self.count1, '/', len_batch * self.local_ep)
                        dfx, net_glob_server = train_server(client_fx, labels, iter, self.local_ep, self.idx,
                                                            len_batch, self.net_glob_server,
                                                            self.lr, self.criterion, self.batch_acc_train,
                                                            self.batch_loss_train, self.count1,
                                                            self.loss_train_collect_user,
                                                            self.acc_train_collect_user, self.idx_collect, self.idx_disconnected, self.idx_round_disconnected,
                                                            self.num_users)

                        if dfx is None:
                            # server returned invalid gradient
                            print(f"[Warning] Server returned invalid gradient for client {self.idx}")
                            break

                        fx.backward(dfx)
                        for param in net.parameters():
                            assert param.grad.dtype == torch.float, "Gradient type is not float"
                        optimizer_client.step()

                net.to('cpu')
                net_glob_server.to('cpu')

                for param in net.parameters():
                    if torch.isnan(param).any():
                        print(f"[Error] Client{self.idx} 模型参数包含 NaN")

                return net.cpu().state_dict(), self.net_glob_server.cpu().state_dict()
            finally:
                self.status = "idle"  # 任务结束更新状态

    def evaluate(self, w_client, ell):
        if self.is_disconnected:
            print(f"[Skip] Client{self.idx} 断开，跳过测试")
            return
        else:
            try:
                self.status = "testing"  # 更新状态
                net = ResNet50_client_side().cpu()  # 初始化新模型
                net.load_state_dict(w_client)  # 加载训练后的参数
                net = net.to('cuda:0')  # 移到 GPU
                net.eval()
                for param in net.parameters():
                    if torch.isnan(param).any():
                        print(f"[Error] Client{self.idx} 模型参数包含NaN，跳过测试")
                        return

                with torch.no_grad():
                    for images, labels in self.ldr_test:
                        if torch.isnan(images).any() or torch.isnan(labels).any():
                            print(f"[Error] Client{self.idx} 测试数据包含NaN")
                            return
                    len_batch = len(self.ldr_test)
                    for batch_idx, (images, labels) in enumerate(self.ldr_test):

                        if self.is_disconnected:
                            print(f"[Abort] Client{self.idx} 测试期间断开，终止测试")
                            break

                        images, labels = images.to('cuda:0'), labels.to('cuda:0')
                        fx = net(images)
                        evaluate_server(fx, labels, self.idx, len_batch, ell)

            finally:
                self.status = "idle"  # 任务结束
                net.to('cpu')


# =====================================================================================================
# dataset_iid(), dataset_non_iid(), monitor_heartbeats(), draw_data_distribution()
# 保持与原脚本一致
# =====================================================================================================

def dataset_iid(dataset, num_users):
    labels = [label for _, label in dataset]
    class_idxs = {i: [] for i in range(10)}
    for idx, label in enumerate(labels):
        class_idxs[label].append(idx)

    dict_users = {}
    num_per_class = len(class_idxs[0]) // num_users  # 假设每个类别的样本数相同

    for user in range(num_users):
        dict_users[user] = set()
        for class_idx in class_idxs:
            start = user * num_per_class
            end = (user + 1) * num_per_class
            dict_users[user].update(class_idxs[class_idx][start:end])

    return dict_users


def dataset_non_iid(dataset, num_users, class_distribution):
    if len(class_distribution) != num_users or any(len(cd) != 10 for cd in class_distribution):
        raise ValueError("类别分布列表的长度必须等于客户端数量，且每个子列表长度必须为10。")

    labels = [label for _, label in dataset]
    class_idxs = {i: [] for i in range(10)}
    for idx, label in enumerate(labels):
        class_idxs[label].append(idx)

    dict_users = {i: set() for i in range(num_users)}
    used_indices = {i: [] for i in range(10)}

    for user in range(num_users):
        for cls in range(10):
            if class_distribution[user][cls] == 1:
                class_indices = class_idxs[cls]
                half_len = len(class_indices) // 2
                available_indices = [idx for idx in class_indices if idx not in used_indices[cls]]
                if len(available_indices) >= half_len:
                    assigned_indices = available_indices[:half_len]
                    dict_users[user].update(assigned_indices)
                    used_indices[cls].extend(assigned_indices)
                else:
                    dict_users[user].update(available_indices)
                    used_indices[cls].extend(available_indices)

    return dict_users

def cifar_user_dataset(dataset, num_users, noniid_fraction):
    """
    Create a split similar to v2_cifar10.py: a fraction of the dataset is non-iid (sharded by label)
    and the rest is IID across users.
    noniid_fraction is a value in [0,1] indicating portion of samples assigned in a non-iid way.
    """
    # initialization
    total_items = len(dataset)
    num_noniid_items = int(total_items * noniid_fraction)
    num_iid_items = total_items - num_noniid_items
    dict_users = [list() for _ in range(num_users)]
    idxs = [i for i in range(len(dataset))]

    # IID portion
    if num_iid_items != 0:
        per_user_iid_items = int(num_iid_items / num_users)
        for ii in range(num_users):
            tmp_set = set(np.random.choice(idxs, per_user_iid_items, replace=False))
            dict_users[ii] += list(tmp_set)
            idxs = list(set(idxs) - tmp_set)

    # NON-IID portion: shard by label and assign shards to users
    if num_noniid_items != 0:
        num_shards = num_users
        per_shards_num_imgs = int(num_noniid_items / num_shards)
        # collect labels for remaining idxs
        labels = [dataset[i][1] for i in idxs]
        idxs = np.array(idxs)
        labels = np.array(labels)
        # sort by label
        order = labels.argsort()
        idxs = idxs[order]

        idx_shard = [i for i in range(num_shards)]
        i = 0
        while idx_shard:
            rand_idx = np.random.choice(idx_shard, 1, replace=False)[0]
            idx_shard = list(set(idx_shard) - {rand_idx})
            start = int(rand_idx) * per_shards_num_imgs
            end = (int(rand_idx) + 1) * per_shards_num_imgs
            selected = list(idxs[start:end])
            dict_users[i].extend(selected)
            i = (i + 1) % num_users

    # convert to dict of sets to match other utilities
    return {i: set(dict_users[i]) for i in range(num_users)}

import numpy as np
from collections import defaultdict

def cifar_user_dataset_dirichlet(dataset, num_users, noniid_fraction, alpha=0.1, balanced=True, seed=None):
    """
    CIFAR dataset split using Dirichlet distribution (α controls non-IID level).

    Args:
        dataset: PyTorch dataset (each item -> (data, label))
        num_users: int, number of users
        noniid_fraction: float ∈ [0,1], portion of dataset assigned non-IID
        alpha: float, Dirichlet concentration parameter (smaller => stronger non-IID)
        balanced: bool, if True each user has same # of samples (data-count balanced)
        seed: int or None, random seed for reproducibility
    Returns:
        dict_users: {user_id: set(sample_indices)} — same output format as original
    """
    if seed is not None:
        np.random.seed(seed)

    total_items = len(dataset)
    num_noniid_items = int(total_items * noniid_fraction)
    num_iid_items = total_items - num_noniid_items
    all_indices = np.arange(total_items)

    dict_users = [set() for _ in range(num_users)]

    # -------------------- #
    # 1️⃣ IID 部分
    # -------------------- #
    if num_iid_items > 0:
        iid_indices = np.random.choice(all_indices, num_iid_items, replace=False)
        all_indices = np.setdiff1d(all_indices, iid_indices)
        per_user_iid = num_iid_items // num_users
        for i in range(num_users):
            chosen = iid_indices[i * per_user_iid : (i + 1) * per_user_iid]
            dict_users[i].update(chosen)

    # -------------------- #
    # 2️⃣ 非IID部分 (Dirichlet)
    # -------------------- #
    if num_noniid_items > 0:
        noniid_indices = all_indices
        labels = np.array([dataset[i][1] for i in noniid_indices])
        num_classes = len(np.unique(labels))

        class_indices = {c: np.where(labels == c)[0] for c in np.unique(labels)}
        class_proportions = np.random.dirichlet([alpha] * num_users, size=num_classes)

        user_data = defaultdict(list)
        for c, idxs in class_indices.items():
            np.random.shuffle(idxs)
            props = class_proportions[c]
            class_split = (np.cumsum(props) * len(idxs)).astype(int)
            split_indices = np.split(idxs, class_split[:-1])
            for u, idxs_u in enumerate(split_indices):
                user_data[u].extend(noniid_indices[idxs_u])

        for u in range(num_users):
            dict_users[u].update(user_data[u])

    # -------------------- #
    # 3️⃣ 平衡处理（无数据丢失）
    # -------------------- #
    if balanced:
        total_per_user = total_items // num_users
        all_used = set()
        extra_pool = []

        # Step 1: 裁剪过多样本并收集多余的
        for u in range(num_users):
            data_u = list(dict_users[u])
            if len(data_u) > total_per_user:
                keep = np.random.choice(data_u, total_per_user, replace=False)
                extra = list(set(data_u) - set(keep))
                dict_users[u] = set(keep)
                extra_pool.extend(extra)
            all_used.update(dict_users[u])

        # Step 2: 构建补齐池（包含未用样本 + 裁剪样本）
        remaining = list(set(np.arange(total_items)) - all_used)
        remaining.extend(extra_pool)
        np.random.shuffle(remaining)

        # Step 3: 补齐不足用户（无重复）
        ptr = 0
        for u in range(num_users):
            need = total_per_user - len(dict_users[u])
            if need > 0:
                add_samples = remaining[ptr: ptr + need]
                dict_users[u].update(add_samples)
                ptr += need

    # -------------------- #
    # 4️⃣ 输出
    # -------------------- #
    return {i: set(dict_users[i]) for i in range(num_users)}




def monitor_heartbeats(heartbeat_queue, num_users):
    client_status = {i: {"status": "idle", "last_heartbeat": "", "type": "normal"} for i in range(num_users)}
    while True:
        try:
            idx, status, timestamp = heartbeat_queue.get(timeout=30)
            client_status[idx] = {"status": status, "last_heartbeat": timestamp, "type": "normal" if status != "disconnected" else "disconnected"}
        except queue.Empty:
            for idx in range(num_users):
                if client_status[idx]["type"] !=  "disconnected" and client_status[idx]["status"] in ["training", "testing"]:
                    print(f"[Error] Client {idx} exited unexpectedly!")
                    client_status[idx] = {"status": "idle", "last_heartbeat": "", "type": "disconnected"}
        except IOError as e:
            if "[WinError 232]" in str(e):
                print("[Info] 管道正常关闭，退出监测...")
                return
        except Exception as e:
            print(f"[Error] An unexpected error occurred: {e}")


def cleanup_client(local):
    local.stop_heartbeat()
    del local


def draw_data_distribution(dict_users, dataset, num_users, save_path='data_distribution.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    client_dist = {i: [0]*10 for i in range(num_users)}
    for client_idx, indices in dict_users.items():
        labels = [dataset[idx][1] for idx in indices]
        for label in labels:
            client_dist[client_idx][label] += 1

    fig, axes = plt.subplots(nrows=num_users, ncols=1, figsize=(12, 3*num_users))
    for i in range(num_users):
        ax = axes[i]
        ax.bar(range(10), client_dist[i], color='skyblue')
        ax.set_title(f'Client {i} Data Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(range(10))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


# ====================== 新增：连续动作的策略网络（Actor）实现 REINFORCE ======================
class ActorContinuous(nn.Module):
    """
    输出一个动作 mean，然后从 N(mean, sigma^2) 中采样动作，并通过 sigmoid 映射到 [0,1] 作为 correction_rate。
    使用 REINFORCE（one-step）更新策略。
    """
    def __init__(self, state_dim=3, hidden=64, lr=1e-3, init_log_std=-1.0):
        super(ActorContinuous, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)  # 输出 mean
        )
        # 将 log_std 作为可学习参数或常量；这里使用可学习参数
        self.log_std = nn.Parameter(torch.tensor(init_log_std))
        self.optimizer = torch.optim.Adam(list(self.net.parameters()) + [self.log_std], lr=lr)

    def forward(self, state):
        # state: torch tensor shape (state_dim,)
        mean = self.net(state)  # shape (1,)
        std = torch.exp(self.log_std) + 1e-6
        return mean.squeeze(), std

    def select_action(self, state, deterministic=False):
        """
        返回 correction_rate (float in [0,1]) 和 log_prob (torch scalar)
        """
        mean, std = self.forward(state)
        if deterministic:
            action_raw = mean
            log_prob = None
        else:
            # 从 normal 分布采样
            noise = torch.normal(mean=torch.zeros_like(mean), std=std)
            action_raw = mean + noise
            var = std ** 2
            # log prob under Normal(mean, std)
            # for scalar: log_prob = -0.5*((x-mean)^2/var + log(2pi var))
            log_prob = -0.5 * ((action_raw - mean) ** 2 / var + torch.log(2 * torch.pi * var))
        # map to [0,1]
        action = torch.sigmoid(action_raw)
        return action.item(), log_prob

    def update(self, log_prob, advantage):
        """
        REINFORCE update: maximize log_prob * advantage -> minimize -log_prob * advantage
        """
        if log_prob is None:
            return
        loss = -log_prob * advantage  # negative for gradient ascent
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


if __name__ == '__main__':
    torch.cuda.init()
    torch.multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    running = manager.Value('b', True)

    parser = argparse.ArgumentParser(description='Training script with RL correction_rate')
    parser.add_argument('--epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--disconnect_prob', type=float, default=0.40, help='Disconnect probability')
    parser.add_argument('--disconnect_round', type=int, default=3, help='Disconnect round')
    parser.add_argument("--correction_rate", type=float, default=0.5, help="Initial Correction rate (will be overridden by RL)")
    parser.add_argument("--local_ep", type=int, default=3, help="Local epochs")
    parser.add_argument('--lr_decay', type=float, default=1, help='Learning rate decay factor')
    parser.add_argument("--lr", type=float, default=0.001, help='Learning rate')
    parser.add_argument("--rl_lr", type=float, default=1e-3, help="RL agent learning rate")
    parser.add_argument("--alpha_reward", type=float, default=0.5, help="Reward loss penalty coefficient")
    parser.add_argument("--noniid_fraction", type=float, default=1.0, help="Fraction of non-IID data distribution (0~1)")
    args = parser.parse_args()

    SEED = 1234
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))

    available_gpus = [i for i in range(torch.cuda.device_count())]
    print(f"Available GPUs: {available_gpus}")

    program = "PipeSFLV1 ResNet50 on CIFAR-10 with RL correction_rate"
    print(f"---------{program}----------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # ===================================================================
    # No. of users
    num_users = 4
    epochs = args.epochs
    disconnect_prob = args.disconnect_prob
    disconnect_round = args.disconnect_round
    correction_rate = args.correction_rate
    local_ep = args.local_ep
    frac = 1  # participation of clients; if 1 then 100% clients participate in SFLV2
    noniid_fraction = args.noniid_fraction if hasattr(args, 'noniid_fraction') else 1.0
    lr = args.lr
    lr_decay = args.lr_decay
    train_times = []

    net_glob_client = ResNet50_client_side().cpu()
    print(net_glob_client)

    net_glob_server = ResNet50_server_side(10).cpu()
    print(net_glob_server)

    prev_w_glob_client = {k: v.cpu() for k, v in copy.deepcopy(net_glob_client.state_dict()).items()}
    prev_w_glob_server = {k: v.cpu() for k, v in copy.deepcopy(net_glob_server.state_dict()).items()}

    # ===================================================================================
    # For Server Side Loss and Accuracy
    loss_train_collect = manager.list()
    acc_train_collect = manager.list()
    loss_test_collect = manager.list()
    acc_test_collect = manager.list()
    batch_acc_test = manager.list()
    batch_loss_test = manager.list()

    criterion = nn.CrossEntropyLoss()
    count1 = 0
    count2 = 0

    loss_test_collect_user = manager.list()
    acc_test_collect_user = manager.list()

    # client idx collector
    idx_collect = manager.list()
    idx_disconnected = manager.list()
    idx_round_disconnected = manager.list()

    idx_disconnected_time = manager.list([0] * num_users)  # 初始化倒计时列表

    l_epoch_check = False
    fed_check = False

    heartbeat_queue = manager.Queue()
    monitor_process = multiprocessing.Process(target=monitor_heartbeats, args=(heartbeat_queue, num_users))
    monitor_process.start()

    # 初始化校正变量
    client_corrections = {i: {k: torch.zeros_like(v) for k, v in net_glob_client.state_dict().items()} for i in
                          range(num_users)}
    server_corrections = {i: {k: torch.zeros_like(v) for k, v in net_glob_server.state_dict().items()} for i in
                          range(num_users)}

    # =============================================================================
    #                         Data preprocessing
    # =============================================================================

    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2023, 0.1994, 0.2010]

    train_transforms = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    test_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

    dataset_train = datasets.CIFAR10(
        root='./data/cifar10',
        train=True,
        download=True,
        transform=train_transforms
    )

    dataset_test = datasets.CIFAR10(
        root='./data/cifar10',
        train=False,
        download=True,
        transform=test_transforms
    )

    dict_users = cifar_user_dataset_dirichlet(dataset_train, num_users, noniid_fraction=noniid_fraction, alpha=0.2, balanced=False, seed=27)
    dict_users_test = dataset_iid(dataset_test, num_users)
    draw_data_distribution(dict_users, dataset_train, num_users,
                           save_path='output/data_distribution.png')
    # 输出每个客户端的数据量（用于加权聚合的权重）——仅输出一次，便于检查数据划分
    try:
        client_sample_counts = [len(dict_users[i]) for i in range(num_users)]
        total_samples = sum(client_sample_counts)
        print(f"[Info] Client sample counts: {client_sample_counts}")
        print(f"[Info] Total samples: {total_samples}")
        if total_samples > 0:
            normalized = [c / total_samples for c in client_sample_counts]
        else:
            normalized = [0 for _ in client_sample_counts]
        print(f"[Info] Normalized aggregation weights: {normalized}")
    except Exception as e:
        print(f"[Warning] Failed to compute client sample counts: {e}")

    net_glob_client.train()
    w_glob_client = net_glob_client.state_dict()

    # ========================== 初始化 RL Agent ==========================
    state_dim = 3
    agent = ActorContinuous(state_dim=state_dim, hidden=64, lr=args.rl_lr, init_log_std=-1.0)
    running_reward = 0.0  # moving average baseline for advantage
    running_reward_alpha = 0.05  # baseline update rate
    alpha_reward = args.alpha_reward  # loss penalty coefficient

    # 初始化历史指标
    prev_acc = 0.0
    prev_loss = 1.0
    prev_state = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float)

    for iter in range(epochs):
        idx_collect[:] = []
        idx_round_disconnected[:] = []
        start_time = time.time()
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        w_locals_client = []
        w_glob_server_buffer = []
        w_locals_weights = []

        global_seed = iter  # 可自定义种子值
        numpy.random.seed(global_seed)

        running.value = True

        if len(idx_disconnected) > 0 and len(idx_disconnected) < num_users:
            print(f"[sort idxs_users] sort idxs_users")
            idxs_users = sorted(idxs_users, key=lambda x: x not in idx_disconnected)
            print(f"[Round {iter}] Sorted idxs_users: {idxs_users}")

        # ===== RL: 在该轮开始前由 agent 给出 correction_rate（可以基于上一轮状态） =====
        # 如果希望让 agent 根据上一轮的表现做决策，则使用 prev_state；此处我们使用 prev_state
        corr_action, corr_logprob = agent.select_action(prev_state)
        correction_rate = float(corr_action)
        # corr_logprob 可能为 scalar tensor (如果 deterministic=False)
        # 打印选择
        print(f"[RL] Round {iter} initial correction_rate = {correction_rate:.4f}")

        for idx in idxs_users:
            print(f"[Round {iter}] Current user's idx: {idx}")
            if idx in idx_disconnected:
                local = Client(net_glob_client, idx, lr, net_glob_server, criterion, count1, idx_collect, num_users,
                               dataset_train=dataset_train,
                               dataset_test=dataset_test, idxs=dict_users[idx], idxs_test=dict_users_test[idx],
                               heartbeat_queue=heartbeat_queue, disconnect_prob=disconnect_prob,
                               idx_disconnected=idx_disconnected, running=running, is_disconnected=True, idx_disconnected_time=idx_disconnected_time, idx_round_disconnected=idx_round_disconnected, disconnect_seed=global_seed, disconnect_round=disconnect_round, local_ep=local_ep)
                if idx not in idx_round_disconnected:
                    idx_round_disconnected.append(idx)
            else:
                local = Client(net_glob_client, idx, lr, net_glob_server, criterion, count1, idx_collect, num_users,
                               dataset_train=dataset_train,
                               dataset_test=dataset_test, idxs=dict_users[idx], idxs_test=dict_users_test[idx],
                               heartbeat_queue=heartbeat_queue, disconnect_prob=disconnect_prob,
                               idx_disconnected=idx_disconnected, running=running, is_disconnected=False, idx_disconnected_time=idx_disconnected_time, idx_round_disconnected=idx_round_disconnected, disconnect_seed=global_seed, disconnect_round=disconnect_round, local_ep=local_ep)

            w_client, w_glob_server = local.train(net=copy.deepcopy(net_glob_client))

            if local.is_disconnected:
                prRed(f"Client{idx} 断开连接，使用校正变量模拟更新")
                offline_w_client = {
                    k: (1 - correction_rate) * prev_w_glob_client[k] + correction_rate * (prev_w_glob_client[k] - client_corrections[idx].get(k,
                                                                                                                torch.zeros_like(
                                                                                                                    prev_w_glob_client[
                                                                                                                        k])))
                    for k in prev_w_glob_client.keys()
                }
                for k in offline_w_client.keys():
                    assert offline_w_client[k].dtype == torch.float, f"Param {k} type is {offline_w_client[k].dtype}"

                offline_w_glob_server = {
                    k: (1 - correction_rate) * prev_w_glob_server[k] + correction_rate * (prev_w_glob_server[k] - server_corrections[idx].get(k,
                                                                                                                torch.zeros_like(
                                                                                                                    prev_w_glob_server[
                                                                                                                        k])))
                    for k in prev_w_glob_server.keys()
                }
                w_locals_client.append(offline_w_client)
                w_glob_server_buffer.append(offline_w_glob_server)
                # collect sample count for weighted aggregation
                try:
                    sample_count = len(dict_users[idx])
                except Exception:
                    sample_count = 0
                w_locals_weights.append(sample_count)
            else:
                w_locals_client.append(w_client)  # 已在 CPU
                w_glob_server_buffer.append(w_glob_server)  # 已在 CPU
                # collect sample count for weighted aggregation
                try:
                    sample_count = len(dict_users[idx])
                except Exception:
                    sample_count = 0
                w_locals_weights.append(sample_count)

                # 客户端训练后，更新客户端校正项
                global_update_client = net_glob_client.state_dict()
                for k in global_update_client.keys():
                    client_corrections[idx][k] = torch.clamp((global_update_client[k] - w_client[k]).float(), -1e3,
                                                             1e3)

                # 服务器端训练后，更新服务器端校正项
                global_update_server = net_glob_server.state_dict()
                for k in global_update_server.keys():
                    server_corrections[idx][k] = torch.clamp((global_update_server[k] - w_glob_server[k]).float(), -1e3,
                                                             1e3)
                # Testing -------------------
                local.evaluate(w_client, ell=iter)

            cleanup_thread = threading.Thread(target=cleanup_client, args=(local,), daemon=True)
            cleanup_thread.start()

        running.value = False

        prev_w_glob_client = {k: v.cpu() for k, v in copy.deepcopy(net_glob_client.state_dict()).items()}
        prev_w_glob_server = {k: v.cpu() for k, v in copy.deepcopy(net_glob_server.state_dict()).items()}

        print("------------------------------------------------------------")
        print("------ Fed Server: Federation process at Client-Side -------")
        print("------------------------------------------------------------")

        if len(w_locals_client) == 0:
            print("No clients available for Federated Learning!")
        else:
            w_glob_client = FedAvg(w_locals_client, client_corrections, model_type='client', weights=w_locals_weights)
            w_glob_server = FedAvg(w_glob_server_buffer, server_corrections, model_type='server', weights=w_locals_weights)
            net_glob_client.load_state_dict(w_glob_client)
            net_glob_server.load_state_dict(w_glob_server)

        train_time = time.time() - start_time
        train_times.append(train_time)

        print("====================== PipeSFL V1 ========================")
        print('========== Train: Round {:3d} Time: {:2f}s ==============='.format(iter, train_time))
        print("==========================================================")

        # 更新 idx_disconnected_time
        for i in range(len(idx_disconnected_time)):
            if idx_disconnected_time[i] > 0:
                idx_disconnected_time[i] -= 1
                if idx_disconnected_time[i] == 0:
                    print(f"[Reconnect] Client{i} 将在下一轮重新连接")
                    idx_disconnected.remove(i)
                else:
                    print(f"[Reconnect] Client{i} 重新连接倒计时: {idx_disconnected_time[i]}")

        fed_check = False
        print(f"[Debug] len(acc_test_collect): {len(acc_test_collect)}")
        if len(acc_train_collect) > 0:
            if len(acc_train_collect) < iter + 1:
                acc_train_collect.append(acc_train_collect[-1])
                loss_train_collect.append(loss_train_collect[-1])
                print(f"[Debug] acc_train_collect 异常 触发保护机制 重复添加最后一个元素")
        if len(acc_test_collect) > 0:
            if len(acc_test_collect) < len(acc_train_collect):
                acc_test_collect.append(acc_test_collect[-1])
                loss_test_collect.append(loss_test_collect[-1])
                print(f"[Debug] acc_test_collect 异常 触发保护机制 重复添加最后一个元素")
        print("==========================================================")

        # ================== RL: 计算 reward 并更新 agent ==================
        # 使用全局 train 指标作为 RL 输入。注意这些全局变量在 train_server 中被更新（当一轮所有客户端完成时）。
        current_acc = float(acc_avg_all_user_train_global) if acc_avg_all_user_train_global is not None else 0.0
        current_loss = float(loss_avg_all_user_train_global) if loss_avg_all_user_train_global is not None else prev_loss

        disconnect_rate = len(idx_disconnected) / num_users
        current_state = torch.tensor([current_acc / 100.0, current_loss, disconnect_rate], dtype=torch.float)

        # reward: accuracy increase minus alpha * loss increase
        reward = (current_acc - prev_acc) / 100.0 - alpha_reward * (current_loss - prev_loss)

        # moving average baseline
        running_reward = running_reward * (1 - running_reward_alpha) + running_reward_alpha * reward
        advantage = reward - running_reward

        # corr_logprob corresponds to the action selected at the start of the round
        # if corr_logprob is None (deterministic), skip update
        if 'corr_logprob' in locals() and corr_logprob is not None:
            agent.update(corr_logprob, advantage)

        # 打印 RL 相关信息
        print(f"[RL] Round {iter} reward={reward:.4f} adv={advantage:.4f} running_baseline={running_reward:.4f}")
        print(f"[RL] Round {iter} prev_acc={prev_acc:.3f} acc={current_acc:.3f} prev_loss={prev_loss:.4f} loss={current_loss:.4f}")

        # 下一轮使用当前状态来选择动作（也可在下一轮开始时再选）
        next_action, next_logprob = agent.select_action(current_state)
        # 更新 prev_state, prev_acc, prev_loss
        prev_state = current_state
        prev_acc = current_acc
        prev_loss = current_loss
        # 将下轮的选择赋给 corr_logprob 以便下一次 update 使用
        corr_logprob = next_logprob

        # decay lr
        lr = lr * lr_decay

    # ===================================================================================

    print("Training and Evaluation completed!")

    # 保存与绘图（保持原逻辑）
    curve_dir = 'output/curve/cifar10/long_offline_rl'
    model_dir = 'output/model/cifar10/long_offline_rl'
    acc_dir = 'output/acc/cifar10/long_offline_rl'
    loss_dir = 'output/loss/cifar10/long_offline_rl'

    for directory in [curve_dir, model_dir, acc_dir, loss_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    plt.plot(range(epochs), train_times)
    plt.xlabel('Training Rounds')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Curve')
    plt.grid(True)
    prefix = f"_CIFAR10_ep{args.epochs}_dp{args.disconnect_prob:.2f}_dr{args.disconnect_round}_rlCR_le{args.local_ep}_lr{args.lr}_noniid{args.noniid_fraction:.2f}"
    curve_filename = os.path.join(curve_dir, f'train_time_curve{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.png')
    plt.savefig(curve_filename)
    plt.clf()

    client_model_filename = os.path.join(model_dir, f'Client{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.pth')
    server_model_filename = os.path.join(model_dir, f'Server{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.pth')

    torch.save(net_glob_client.state_dict(), client_model_filename)
    torch.save(net_glob_server.state_dict(), server_model_filename)
    print('Model saved successfully!')

    acc_train_collect_list = list(acc_train_collect)
    loss_train_collect_list = list(loss_train_collect)
    acc_test_collect_list = list(acc_test_collect)
    loss_test_collect_list = list(loss_test_collect)

    acc_train_df = pd.DataFrame(acc_train_collect_list)
    loss_train_df = pd.DataFrame(loss_train_collect_list)
    acc_test_df = pd.DataFrame(acc_test_collect_list)
    loss_test_df = pd.DataFrame(loss_test_collect_list)

    acc_train_filename = os.path.join(acc_dir, f'Client_Acc_RL{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.csv')
    acc_train_df.to_csv(acc_train_filename, index=False)

    loss_train_filename = os.path.join(loss_dir, f'Client_Loss_RL{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.csv')
    loss_train_df.to_csv(loss_train_filename, index=False)

    acc_test_filename = os.path.join(acc_dir, f'Server_Acc_RL{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.csv')
    acc_test_df.to_csv(acc_test_filename, index=False)

    loss_test_filename = os.path.join(loss_dir, f'Server_Loss_RL{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.csv')
    loss_test_df.to_csv(loss_test_filename, index=False)

    plt.plot(range(epochs), acc_train_collect_list, label='Train Accuracy')
    plt.plot(range(epochs), acc_test_collect_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)

    acc_curve_filename = os.path.join(curve_dir,
                                      f'acc_curve_RL{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.png')
    plt.savefig(acc_curve_filename)
    plt.clf()

    plt.plot(range(epochs), loss_train_collect_list, label='Train Loss')
    plt.plot(range(epochs), loss_test_collect_list, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)

    loss_curve_filename = os.path.join(curve_dir,
                                       f'loss_curve_RL{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.png')
    plt.savefig(loss_curve_filename)
    print('Data saved successfully!')

    monitor_process.terminate()
    monitor_process.join()

    time.sleep(5)
    print("程序正常结束")
