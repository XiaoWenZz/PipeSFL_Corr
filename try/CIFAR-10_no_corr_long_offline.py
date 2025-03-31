import queue
import threading
import argparse

import numpy.random
from matplotlib import pyplot as plt
# =============================================================================
# SplitfedV2 (SFLV2) learning: ResNet18 on CIFAR-10
# CIFAR-10 dataset: Tschandl, P.: The CIFAR-10 dataset, a large collection of multi - source dermatoscopic images of common pigmented skin lesions (2018), doi:10.7910/DVN/DBW86T

# We have three versions of our implementations
# Version1: without using socket and no DP+PixelDP
# Version2: with using socket but no DP+PixelDP
# Version3: without using socket but with DP+PixelDP

# This program is Version1: Single program simulation
# ==============================================================================
import torch
from torch import nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Dataset
import pandas as pd
from sklearn.model_selection import train_test_split
from PIL import Image
from glob import glob
import random
import numpy as np
import os
import time
import multiprocessing

# import matplotlib
# matplotlib.use('Agg')
# import matplotlib.pyplot as plt
import copy

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
        # 激活函数使用LeakyReLU
        # out = torch.relu(self.bn1(self.conv1(x)))
        # out = torch.relu(self.bn2(self.conv2(out)))
        # out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        # out = torch.relu(out)
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
        # out = torch.relu(self.bn1(self.conv1(x)))
        # out = self.layer1(out)
        # return out
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
def FedAvg(w, model_type):
    """
    model_type: 'client' 或 'server'，用于选择对应参数键
    """
    if not w:
        return {}

    # 根据模型类型获取基准参数
    if model_type == 'client':
        w_avg = copy.deepcopy(w[0])
        param_keys = net_glob_client.state_dict().keys()
    elif model_type == 'server':
        w_avg = copy.deepcopy(w[0])
        param_keys = net_glob_server.state_dict().keys()
    else:
        raise ValueError("Invalid model_type")

    if len(w) == 1:
        # 如果只有一个元素，直接返回该元素的深拷贝
        return w_avg

    for k in param_keys:
        total = w_avg[k].clone()
        for i, params in enumerate(w[1:], start=1):
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
    # print('client_fx type:', type(fx_client))
    y = y
    # print('y:', y)
    # print('len(y):', len(y))

    # ---------forward prop-------------
    fx_server = net_glob_server(fx_client)

    # calculate loss
    loss = criterion(fx_server, y)
    # calculate accuracy
    acc = calculate_accuracy(fx_server, y)

    # --------backward prop--------------
    loss.backward()
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    # server-side model net_glob_server is global so it is updated automatically in each pass to this function

    # count1: to track the completion of the local batch associated with one client
    count1 += 1
    # print('count1:', count1, '<===>len_batch:', len_batch)
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

            # we store the last accuracy in the last batch of the epoch and it is not the average of all local epochs
            # this is because we work on the last trained model and its accuracy (not earlier cases)

            # print("accuracy = ", acc_avg_train)
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            # accumulate accuracy and loss for each new user
            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            # collect the id of each new user
            if idx not in idx_collect:
                if idx in idx_disconnected:
                    print(f"[Warning] Client{idx} 在 idx_disconnected 中，可能存在竞态条件")
                else:
                    idx_collect.append(idx)
                # print(idx_collect)

        # for debugging print idxes of idx_collect and idx_disconnected
        print(f"[Debug] idx_collect: {idx_collect}")
        print(f"[Debug] idx_disconnected: {idx_disconnected}")

        # This is to check if all users are served for one round --------------------
        if len(idx_collect) + len(idx_round_disconnected) == num_users:
            fed_check = True  # to evaluate_server function  - to check fed check has hitted
            # all users served for one round ------------------------- output print and update is done in evaluate_server()
            # for nicer display

            # 确保列表中有数据再计算平均
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
            # for debugging print
            print('[train_server] acc_train_collect appended once, current length:', len(acc_train_collect))
            print('[train_server] current idx_collect:', idx_collect)
            print('[train_server] current idx_disconnected:', idx_disconnected)
            loss_train_collect.append(loss_avg_all_user_train)

    # send gradients to the client
    # server_result_queue.put(dfx_client.to('cuda:'+str(idx)))
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
        # ---------forward prop-------------
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

                # Store the last accuracy and loss
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
                # for debugging print
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
                 dataset_train=None, dataset_test=None, idxs=None, idxs_test=None, heartbeat_queue=None, disconnect_prob=0.001, idx_disconnected=None, is_disconnected=False, idx_disconnected_time=None, idx_round_disconnected=None, disconnect_seed=0, disconnect_round=1, local_ep=1):
        self.disconnect_prob = disconnect_prob  # 断开概率
        self.is_disconnected = is_disconnected  # 是否断开
        self.heartbeat_queue = heartbeat_queue
        self.idx = idx
        # self.device = device
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
        # self.selected_clients = []
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=512, shuffle=True)
        self.ldr_test = DataLoader(DatasetSplit(dataset_test, idxs_test), batch_size=512, shuffle=True)
        self.disconnect_seed = disconnect_seed
        self.rng = numpy.random.default_rng(seed=self.disconnect_seed + self.idx)
        self.disconnect_round = disconnect_round

        self.idx_disconnected = idx_disconnected
        self.idx_round_disconnected = idx_round_disconnected
        self.idx_disconnected_time = idx_disconnected_time
        self.running = running
        # 新增心跳管理
        self.status = "idle"  # idle, training, testing
        self.heartbeat_interval =3  # 3秒心跳间隔
        self.stop_heartbeat_flag = False
        # 心跳线程在初始化最后启动（确保属性已创建）
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def send_heartbeat(self):
        try:
            # while self.running.value and not self.stop_heartbeat_flag:
            while not self.stop_heartbeat_flag:
                if not self.is_disconnected and not self.stop_heartbeat_flag:
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
            # for debugging print
            print('[update_fed_check] acc_train_collect appended once, current length:', len(acc_train_collect))
            print('[update_fed_check] current idx_collect:', self.idx_collect)
            print('[update_fed_check] current idx_disconnected:', self.idx_disconnected)
            loss_train_collect.append(loss_avg_all_user_train)
            if len(acc_test_collect) == 0:
                acc_test_collect.append(0)
                loss_test_collect.append(0)
            else:
                acc_test_collect.append(acc_test_collect[-1])
                loss_test_collect.append(loss_test_collect[-1])
            # for debugging print
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

                        fx.backward(dfx)
                        optimizer_client.step()

                net.to('cpu')
                net_glob_server.to('cpu')
                return net.cpu().state_dict(), self.net_glob_server.cpu().state_dict()
            finally:
                self.status = "idle"  # 任务结束更新状态

    def evaluate(self, net, ell):
        if self.is_disconnected:
            print(f"[Skip] Client{self.idx} 断开，跳过测试")
            return
        else:
            try:
                self.status = "testing"  # 更新状态
                net.eval()
                # print(f"[Debug-before-evaluate] Client{self.idx} 测试前参数示例: {list(net.parameters())[0][:2]}")

                with torch.no_grad():
                    len_batch = len(self.ldr_test)
                    for batch_idx, (images, labels) in enumerate(self.ldr_test):

                        # 检查客户端是否断开连接
                        if self.is_disconnected:
                            print(f"[Abort] Client{self.idx} 测试期间断开，终止测试")
                            break

                        images, labels = images.to('cuda:0'), labels.to('cuda:0')
                        fx = net(images)
                        # print([f"[Debug-before-evaluate-server] Client{self.idx} fx 部分输出: {fx[:2]}"])

                        evaluate_server(fx, labels, self.idx, len_batch, ell)

            finally:
                self.status = "idle"  # 任务结束更新状态


# =====================================================================================================
# dataset_iid() will create a dictionary to collect the indices of the data samples randomly for each client
# IID CIFAR-10 datasets will be created based on this
# def dataset_iid(dataset, num_users):
#     num_items = int(len(dataset) / num_users)
#     dict_users, all_idxs = {}, [i for i in range(len(dataset))]
#     for i in range(num_users):
#         dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
#         all_idxs = list(set(all_idxs) - dict_users[i])
#     return dict_users

def dataset_iid(dataset, num_users):
    # 获取数据集中的标签列表
    labels = [label for _, label in dataset]
    # 统计每个类别的样本索引
    class_idxs = {i: [] for i in range(10)}  # CIFAR-10有10个类别
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


def monitor_heartbeats(heartbeat_queue, num_users):
    client_status = {i: {"status": "idle", "last_heartbeat": "", "type": "normal"} for i in range(num_users)}
    while True:
        try:
            idx, status, timestamp = heartbeat_queue.get(timeout=30)
            client_status[idx] = {"status": status, "last_heartbeat": timestamp, "type": "normal" if status != "disconnected" else "disconnected"}
            # print(f"[Heartbeat] Client {idx}: {status} - Last: {timestamp} - ({client_status[idx]['type']})")

            # 检查超时（超过3倍间隔未收到心跳）
            # if (time.time() - time.mktime(
            #         time.strptime(client_status[idx]["last_heartbeat"], "%Y-%m-%d %H:%M:%S"))) > 3 * 10:
            #     print(f"[Warning] Client {idx} may be disconnected!")

        except queue.Empty:
            # 处理意外退出
            for idx in range(num_users):
                # if client_status[idx]["status"] in ["training", "testing"]:
                #     print(f"[Error] Client {idx} exited unexpectedly!")
                #     client_status[idx] = {"status": "idle", "last_heartbeat": time.strftime("%Y-%m-%d %H:%M:%S")}
                if client_status[idx]["type"] !=  "disconnected" and client_status[idx]["status"] in ["training", "testing"]:
                    print(f"[Error] Client {idx} exited unexpectedly!")
                    client_status[idx] = {"status": "idle", "last_heartbeat": "", "type": "disconnected"}

        except IOError as e:  # 捕获管道关闭错误
            if "[WinError 232]" in str(e):
                print("[Info] 管道正常关闭，退出监测...")
                return

        except Exception as e:
            print(f"[Error] An unexpected error occurred: {e}")

def cleanup_client(local):
    local.stop_heartbeat()
    del local

if __name__ == '__main__':
    torch.cuda.init()
    torch.multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    running = manager.Value('b', True)

    parser = argparse.ArgumentParser(description='Training script')
    parser.add_argument('--epochs', type=int, default=20, help='Number of epochs')
    parser.add_argument('--disconnect_prob', type=float, default=0.40, help='Disconnect probability')
    parser.add_argument('--disconnect_round', type=int, default=1, help='Disconnect round')
    parser.add_argument("--local_ep", type=int, default=10, help="Number of local epochs")
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay factor')
    parser.add_argument("--lr", type=int, default=0.0003, help='Learning rate')
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

    # ===================================================================
    program = "PipeSFLV1 ResNet50 on CIFAR-10"
    print(f"---------{program}----------")  # this is to identify the program in the slurm outputs files

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # ===================================================================
    # No. of users
    num_users = 3
    epochs = args.epochs
    disconnect_prob = args.disconnect_prob
    disconnect_round = args.disconnect_round
    local_ep = args.local_ep
    frac = 1  # participation of clients; if 1 then 100% clients participate in SFLV2
    lr = args.lr
    lr_decay = args.lr_decay
    train_times = []

    net_glob_client = ResNet50_client_side().cpu()
    print(net_glob_client)

    net_glob_server = ResNet50_server_side(10).cpu()
    print(net_glob_server)

    # ===================================================================================
    # For Server Side Loss and Accuracy
    loss_train_collect = manager.list()
    acc_train_collect = manager.list()
    loss_test_collect = manager.list()
    acc_test_collect = manager.list()
    # batch_acc_train = manager.list()
    # batch_loss_train = manager.list()
    batch_acc_test = manager.list()
    batch_loss_test = manager.list()

    criterion = nn.CrossEntropyLoss()
    count1 = 0
    count2 = 0

    # to print train - test together in each round-- these are made global
    # acc_avg_all_user_train = 0
    # loss_avg_all_user_train = 0
    # loss_train_collect_user = manager.list()
    # acc_train_collect_user = manager.list()
    loss_test_collect_user = manager.list()
    acc_test_collect_user = manager.list()

    # client idx collector
    idx_collect = manager.list()
    idx_disconnected = manager.list()
    # 当轮内断开的客户端列表 每轮清空 防止fed_check异常导致两次append
    idx_round_disconnected = manager.list()

    # long offline 修改点一 新增数据结构 idx_disconnected_time
    idx_disconnected_time = manager.list([0] * num_users)  # 初始化倒计时列表
    l_epoch_check = False
    fed_check = False

    # 添加心跳队列
    heartbeat_queue = manager.Queue()

    # 启动心跳监测进程
    monitor_process = multiprocessing.Process(target=monitor_heartbeats, args=(heartbeat_queue, num_users))
    monitor_process.start()

    # =============================================================================
    #                         Data preprocessing
    # =============================================================================
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

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

    # 加载CIFAR-10数据集
    dataset_train = datasets.CIFAR10(root='./data', train=True,
                                     download=True, transform=train_transforms)
    dataset_test = datasets.CIFAR10(root='./data', train=False,
                                    download=True, transform=test_transforms)

    # ----------------------------------------------------------------
    dict_users = dataset_iid(dataset_train, num_users)
    dict_users_test = dataset_iid(dataset_test, num_users)

    # ------------ Training And Testing -----------------
    net_glob_client.train()
    # copy weights
    w_glob_client = net_glob_client.state_dict()

    # Federation takes place after certain local epochs in train() client-side
    # this epoch is global epoch, also known as rounds

    for iter in range(epochs):
        # 清空idx_collect和idx_round_disconnected
        idx_collect[:] = []
        idx_round_disconnected[:] = []
        start_time = time.time()
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        w_locals_client = []
        w_glob_server_buffer = []
        global_seed = iter  # 可自定义种子值
        numpy.random.seed(global_seed)

        running.value = True

        # 若idx_disconnected中有客户端且不是全部客户端都断开，则为idx_users排序，确保未断开的客户端在最后 保证eval_server时不会出错
        # 排序逻辑 在idx_disconnected中的客户端排在最前
        if len(idx_disconnected) > 0 and len(idx_disconnected) < num_users:
            print(f"[sort idxs_users] sort idxs_users")
            idxs_users = sorted(idxs_users, key=lambda x: x not in idx_disconnected)
            print(f"[Round {iter}] Sorted idxs_users: {idxs_users}")

        for idx in idxs_users:
            # for debugging print
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

            # Training ------------------
            w_client, w_glob_server = local.train(net=copy.deepcopy(net_glob_client))

            if local.is_disconnected:
                prRed(f"Client{idx} 断开连接，不使用校正变量模拟更新，直接跳过")
                continue
            else:
                w_locals_client.append(w_client)  # 已在 CPU
                w_glob_server_buffer.append(w_glob_server)  # 已在 CPU

                # Testing -------------------
                local.evaluate(net=copy.deepcopy(net_glob_client).to('cuda:0'), ell=iter)

            # 新增：停止当前客户端心跳
            # 在创建客户端后，使用线程执行清理操作
            cleanup_thread = threading.Thread(target=cleanup_client, args=(local,), daemon=True)
            cleanup_thread.start()

        running.value = False

        # Federation process at Client-Side------------------------
        print("------------------------------------------------------------")
        print("------ Fed Server: Federation process at Client-Side -------")
        print("------------------------------------------------------------")

        if len(w_locals_client) == 0:
            print("No clients available for Federated Learning!")

        else:
            # 客户端联邦平均
            w_glob_client = FedAvg(w_locals_client, model_type='client')

            # 服务器端联邦平均
            w_glob_server = FedAvg(w_glob_server_buffer, model_type='server')

            # Update client-side global model
            net_glob_client.load_state_dict(w_glob_client)

            # Update server-side global model
            net_glob_server.load_state_dict(w_glob_server)

        train_time = time.time() - start_time  # 新增：计算当前轮次的训练时间
        train_times.append(train_time)  # 新增：将当前轮次的训练时间添加到列表中

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

        # debug
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

    # ===================================================================================

    print("Training and Evaluation completed!")

    # 确保输出目录存在
    curve_dir = 'output/curve/cifar/long_offline'
    model_dir = 'output/model/cifar/long_offline'
    acc_dir = 'output/acc/cifar/long_offline'
    loss_dir = 'output/loss/cifar/long_offline'

    for directory in [curve_dir, model_dir, acc_dir, loss_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 绘制训练时间曲线
    plt.plot(range(epochs), train_times)
    plt.xlabel('Training Rounds')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Curve')
    plt.grid(True)
    prefix = f"_ep{args.epochs}_dp{args.disconnect_prob:.2f}_dr{args.disconnect_round}_le{args.local_ep}"
    # 保存图片 按照当前时间保存 目录为 output/curve
    curve_filename = os.path.join(curve_dir, f'train_time_curve{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                          time.localtime()) + '.png')
    plt.savefig(curve_filename)
    plt.clf()  # 清除当前图形

    client_model_filename = os.path.join(model_dir,
                                         f'Client{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.pth')
    server_model_filename = os.path.join(model_dir,
                                         f'Server{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.pth')

    torch.save(net_glob_client.state_dict(), client_model_filename)
    torch.save(net_glob_server.state_dict(), server_model_filename)
    print('Model saved successfully!')

    print('length of acc_train_collect:', len(acc_train_collect))
    print('length of loss_train_collect:', len(loss_train_collect))
    # 保存acc和loss数据
    acc_train_collect_list = list(acc_train_collect)
    loss_train_collect_list = list(loss_train_collect)
    acc_test_collect_list = list(acc_test_collect)
    loss_test_collect_list = list(loss_test_collect)

    acc_train_df = pd.DataFrame(acc_train_collect_list)
    loss_train_df = pd.DataFrame(loss_train_collect_list)
    acc_test_df = pd.DataFrame(acc_test_collect_list)
    loss_test_df = pd.DataFrame(loss_test_collect_list)

    acc_train_filename = os.path.join(acc_dir, f'Client_Acc_no_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                           time.localtime()) + '.csv')
    acc_train_df.to_csv(acc_train_filename, index=False)

    loss_train_filename = os.path.join(loss_dir, f'Client_Loss_no_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                              time.localtime()) + '.csv')
    loss_train_df.to_csv(loss_train_filename, index=False)
    # 命名为 模型名+ 数据名+当前时间 目录为 output/acc
    # acc_test_filename = os.path.join(acc_dir, f'Server_Acc_Corr_ep{args.epochs}_dp{args.disconnect_prob:.2f}_dr{args.disconnect_round}_' + time.strftime("%Y%m%d-%H%M%S",
    #                                                                                                    time.localtime()) + '.csv')
    # 使用prefix
    acc_test_filename = os.path.join(acc_dir, f'Server_Acc_no_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                          time.localtime()) + '.csv')
    acc_test_df.to_csv(acc_test_filename, index=False)

    loss_test_filename = os.path.join(loss_dir, f'Server_Loss_no_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                             time.localtime()) + '.csv')
    loss_test_df.to_csv(loss_test_filename, index=False)

    # 绘制训练和测试的acc曲线
    plt.plot(range(epochs), acc_train_collect_list, label='Train Accuracy')
    plt.plot(range(epochs), acc_test_collect_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)

    acc_curve_filename = os.path.join(curve_dir,
                                      f'acc_curve_no_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                 time.localtime()) + '.png')
    plt.savefig(acc_curve_filename)
    plt.clf()  # 清除当前图形

    # 绘制训练和测试的loss曲线
    plt.plot(range(epochs), loss_train_collect_list, label='Train Loss')
    plt.plot(range(epochs), loss_test_collect_list, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)

    loss_curve_filename = os.path.join(curve_dir,
                                       f'loss_curve_no_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                   time.localtime()) + '.png')
    plt.savefig(loss_curve_filename)
    print('Data saved successfully!')

    # 结束心跳监测进程
    monitor_process.terminate()
    monitor_process.join()

    time.sleep(5)  # 等待一段时间，确保所有线程有足够时间退出
    print("程序正常结束")
