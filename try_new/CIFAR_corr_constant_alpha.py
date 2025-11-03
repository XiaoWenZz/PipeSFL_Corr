import queue
import threading
import argparse
from collections import deque

import numpy.random
from matplotlib import pyplot as plt
import torch
from torch import nn, optim
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

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 彩色打印函数
def prRed(skk):
    print("\033[91m {}\033[00m".format(skk))


def prGreen(skk):
    print("\033[92m {}\033[00m".format(skk))


# =====================================================================================================
#                           客户端模型定义
# =====================================================================================================
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
#                           服务端模型定义
# =====================================================================================================
class ResNet50_server_side(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50_server_side, self).__init__()
        self.in_planes = 256  # 经过layer1后的输入通道数

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
#                                  服务端程序
# ====================================================================================================
def FedAvg(w, corrections, model_type):
    if not w:
        return {}

    w = [{k: v.float() for k, v in params.items()} for params in w]

    if model_type == 'client':
        w_avg = copy.deepcopy(w[0])
        param_keys = net_glob_client.state_dict().keys()
    elif model_type == 'server':
        w_avg = copy.deepcopy(w[0])
        param_keys = net_glob_server.state_dict().keys()
    else:
        raise ValueError("Invalid model_type")

    if len(w) == 1:
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


def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, lr, criterion,
                 batch_acc_train, batch_loss_train, count1, loss_train_collect_user, acc_train_collect_user,
                 idx_collect, idx_disconnected, idx_round_disconnected,
                 num_users):
    """Train the global server model on the received activations and return gradients to client.
    This uses the single shared global `net_glob_server` (in-place update), matching v2 behavior.
    """
    global l_epoch_check, fed_check, net_glob_server
    net_glob_server = net_glob_server.to('cuda:0')
    net_glob_server.train()
    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)

    optimizer_server.zero_grad()
    fx_client = fx_client.requires_grad_(True)
    y = y

    fx_server = net_glob_server(fx_client)
    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)

    loss.backward()
    if torch.isnan(fx_client.grad).any():
        print(f"[Error] Server 梯度包含NaN，Client {idx} 的参数将被标记为无效")
        idx_round_disconnected.append(idx)
        return None
    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    count1 += 1
    if count1 == len_batch * l_epoch:
        acc_avg_train = sum(batch_acc_train) / len(batch_acc_train)
        loss_avg_train = sum(batch_loss_train) / len(batch_loss_train)

        batch_acc_train = []
        batch_loss_train = []
        count1 = 0

        prRed('Client{} Train => Local Epoch: {} \tAcc: {:.3f} \tLoss: {:.4f}'.format(idx, l_epoch_count, acc_avg_train,
                                                                                      loss_avg_train))

        if l_epoch_count == l_epoch - 1:
            l_epoch_check = True
            acc_avg_train_all = acc_avg_train
            loss_avg_train_all = loss_avg_train

            loss_train_collect_user.append(loss_avg_train_all)
            acc_train_collect_user.append(acc_avg_train_all)

            if idx not in idx_collect:
                if idx in idx_disconnected:
                    print(f"[Warning] Client{idx} 在 idx_disconnected 中，可能存在竞态条件")
                else:
                    idx_collect.append(idx)

        if len(idx_collect) + len(idx_round_disconnected) == num_users:
            fed_check = True
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
            loss_train_collect.append(loss_avg_all_user_train)

    return dfx_client


def evaluate_server(fx_client, y, idx, len_batch, ell):
    global net_glob_server, criterion, batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect, count2, num_users, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user, acc_avg_all_user_train_global, loss_avg_all_user_train_global
    net_glob_server = net_glob_server.to('cuda:0')
    net_glob_server.eval()

    with torch.no_grad():
        fx_client = fx_client.to('cuda:0')
        y = y.to('cuda:0')

        fx_server = net_glob_server(fx_client)
        loss = criterion(fx_server, y)
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

            if l_epoch_check:
                l_epoch_check = False
                acc_avg_test_all = acc_avg_test
                loss_avg_test_all = loss_avg_test

                loss_test_collect_user.append(loss_avg_test_all)
                acc_test_collect_user.append(acc_avg_test_all)

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


acc_avg_all_user_train_global = 0
loss_avg_all_user_train_global = 0


# ==============================================================================================================
#                                       客户端程序
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


class Client(object):
    def __init__(self, net_client_model, idx, lr, net_glob_server, criterion, count1, idx_collect, num_users, running,
                 dataset_train=None, dataset_test=None, idxs=None, idxs_test=None, heartbeat_queue=None,
                 disconnect_prob=0.001, idx_disconnected=None, is_disconnected=False, idx_disconnected_time=None,
                 idx_round_disconnected=None, disconnect_seed=0, disconnect_round=1, local_ep=1):
        self.disconnect_prob = disconnect_prob
        self.is_disconnected = is_disconnected
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
        self.status = "idle"
        self.heartbeat_interval = 10
        self.stop_heartbeat_flag = False
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def send_heartbeat(self):
        try:
            while not self.stop_heartbeat_flag:
                if not self.is_disconnected:
                    random_num = self.rng.random()
                    self.rng = numpy.random.default_rng(seed=self.disconnect_seed + self.idx)
                    print(f"[send_heartbeat] Client{self.idx} 随机数: {random_num}")
                    self.is_disconnected = random_num < self.disconnect_prob and self.status == "training"
                    if self.is_disconnected:
                        print(f"[Random] Client{self.idx} 随机数: {random_num}")
                        print(f"[Disconnect] Client{self.idx} 断开 (概率{self.disconnect_prob * 100}%)")
                        self.heartbeat_queue.put((self.idx, "disconnected", time.strftime("%Y-%m-%d %H:%M:%S")))

                        if self.idx not in self.idx_disconnected:
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
                    time.sleep(self.heartbeat_interval)
                    continue

                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.heartbeat_queue.put((self.idx, self.status, timestamp))
                time.sleep(self.heartbeat_interval)
        except (EOFError, BrokenPipeError):
            print(f"[Info] Client{self.idx} 心跳线程因连接断开而退出")
        print(f"[Info] Client{self.idx} 心跳线程结束")

    def stop_heartbeat(self):
        self.stop_heartbeat_flag = True
        if self.heartbeat_thread.is_alive():
            self.heartbeat_thread.join()

    def update_fed_check(self):
        global l_epoch_check, fed_check
        if len(self.idx_collect) + len(self.idx_round_disconnected) == self.num_users:
            fed_check = True
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
                self.status = "training"
                net = net.to('cuda:0')
                self.net_glob_server = self.net_glob_server.to('cuda:0')
                net.train()
                optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)

                for iter in range(self.local_ep):
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

                        images, labels = images.to('cuda:0'), labels.to('cuda:0')

                        optimizer_client.zero_grad()
                        fx = net(images)
                        client_fx = fx.clone().detach()

                        client_fx = client_fx.to('cuda:0')
                        self.count1 = self.count1 + 1

                        print('client ', self.idx, ' :', self.count1, '/', len_batch * self.local_ep)
                        dfx = train_server(client_fx, labels, iter, self.local_ep, self.idx,
                                            len_batch, self.lr, self.criterion, self.batch_acc_train,
                                            self.batch_loss_train, self.count1,
                                            self.loss_train_collect_user,
                                            self.acc_train_collect_user, self.idx_collect,
                                            self.idx_disconnected, self.idx_round_disconnected,
                                            self.num_users)

                        fx.backward(dfx)
                        for param in net.parameters():
                            assert param.grad.dtype == torch.float, "Gradient type is not float"
                        optimizer_client.step()

                net.to('cpu')
                # global server lives in module scope; move to cpu for safety
                net_glob_server.to('cpu')

                for param in net.parameters():
                    if torch.isnan(param).any():
                        print(f"[Error] Client{self.idx} 模型参数包含 NaN")

                # Return only client model weights; server is updated in-place globally
                return net.cpu().state_dict(),
            finally:
                self.status = "idle"

    def evaluate(self, w_client, ell):
        if self.is_disconnected:
            print(f"[Skip] Client{self.idx} 断开，跳过测试")
            return
        else:
            try:
                self.status = "testing"
                net = ResNet50_client_side().cpu()
                net.load_state_dict(w_client)
                net = net.to('cuda:0')
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
                self.status = "idle"
                net.to('cpu')


# =====================================================================================================
# 数据集划分函数
# =====================================================================================================
def dataset_iid(dataset, num_users):
    labels = [label for _, label in dataset]
    class_idxs = {i: [] for i in range(10)}
    for idx, label in enumerate(labels):
        class_idxs[label].append(idx)

    dict_users = {}
    num_per_class = len(class_idxs[0]) // num_users

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


def monitor_heartbeats(heartbeat_queue, num_users):
    client_status = {i: {"status": "idle", "last_heartbeat": "", "type": "normal"} for i in range(num_users)}
    while True:
        try:
            idx, status, timestamp = heartbeat_queue.get(timeout=30)
            client_status[idx] = {"status": status, "last_heartbeat": timestamp,
                                  "type": "normal" if status != "disconnected" else "disconnected"}

        except queue.Empty:
            for idx in range(num_users):
                if client_status[idx]["type"] != "disconnected" and client_status[idx]["status"] in ["training",
                                                                                                     "testing"]:
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

    client_dist = {i: [0] * 10 for i in range(num_users)}
    for client_idx, indices in dict_users.items():
        labels = [dataset[idx][1] for idx in indices]
        for label in labels:
            client_dist[client_idx][label] += 1

    fig, axes = plt.subplots(nrows=num_users, ncols=1, figsize=(12, 3 * num_users))
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


# =====================================================================================================
# 强化学习智能体（用于动态决策修正率）
# =====================================================================================================
class RLAgent(nn.Module):
    def __init__(self, state_dim, action_dim=1, lr=0.001, gamma=0.95):
        super(RLAgent, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma

        self.policy_net = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 2 * action_dim)  # 输出均值和标准差
        )

        self.memory = deque(maxlen=10000)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def select_action(self, state):
        """根据状态选择修正率α（带探索的高斯采样）"""
        # 将state转换为张量并移动到模型所在设备
        state = torch.FloatTensor(state).unsqueeze(0).to(self.policy_net[0].weight.device)  # 新增.to(...)
        out = self.policy_net(state)
        mean = out[:, 0]  # α均值
        std = torch.exp(out[:, 1])  # 标准差（确保非负）

        # 限制α范围：0.3~1.5
        mean_clamped = torch.clamp(mean, 0.3, 1.5)
        std_clamped = torch.clamp(std, 0.01, 0.3)  # 限制探索幅度

        # 高斯采样
        action = torch.normal(mean_clamped, std_clamped).item()
        # 最终裁剪到合法范围
        return max(0.3, min(1.5, action))

    def store_transition(self, state, action, reward, next_state):
        """存储经验（s, a, r, s'）"""
        self.memory.append((state, action, reward, next_state))

    def learn(self, batch_size=32):
        """从经验回放中学习，更新策略网络"""
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)

        # move tensors to same device as the policy network
        device = next(self.policy_net.parameters()).device
        states = torch.FloatTensor([s for s, _, _, _ in batch]).to(device)
        actions = torch.FloatTensor([a for _, a, _, _ in batch]).unsqueeze(1).to(device)
        rewards = torch.FloatTensor([r for _, _, r, _ in batch]).unsqueeze(1).to(device)
        next_states = torch.FloatTensor([ns for _, _, _, ns in batch]).to(device)

        # 计算当前状态的动作分布
        out = self.policy_net(states)
        means = out[:, 0].unsqueeze(1)
        stds = torch.exp(out[:, 1]).unsqueeze(1)

        # 避免数值不稳定
        stds = torch.clamp(stds, min=1e-6)

        # 计算动作的对数概率（高斯分布）
        log_probs = -0.5 * ((actions - means) / stds) ** 2 - torch.log(stds * torch.sqrt(torch.tensor(2 * math.pi, device=device)))

        # 奖励函数：以模型性能提升为正反馈
        loss = -torch.mean(log_probs * rewards)  # 策略梯度损失

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()


def get_state(client_idx, offline_time, acc_change, loss_change, data_diff):
    """构建状态特征（归一化到[0,1]）"""
    return [
        min(offline_time / 10, 1.0),  # 离线时长（最大10轮）
        (acc_change + 50) / 100,  # 准确率变化（范围[-50,50]）
        data_diff,  # 数据分布差异（0~1）
        min(loss_change / 2.0, 1.0)  # 损失变化（最大2.0）
    ]


def get_reward(prev_acc, curr_acc, prev_loss, curr_loss):
    """奖励函数：模型性能提升则奖励为正"""
    acc_reward = max(0, curr_acc - prev_acc) * 10  # 准确率提升奖励
    loss_reward = max(0, prev_loss - curr_loss) * 5  # 损失下降奖励
    return acc_reward + loss_reward


# =====================================================================================================
# 主程序
# =====================================================================================================
if __name__ == '__main__':
    torch.cuda.init()
    torch.multiprocessing.set_start_method("spawn", force=True)
    manager = multiprocessing.Manager()
    running = manager.Value('b', True)

    parser = argparse.ArgumentParser(description='Training script with RL-based correction rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--disconnect_prob', type=float, default=0.40, help='Disconnect probability')
    parser.add_argument('--disconnect_round', type=int, default=3, help='Disconnect round')
    parser.add_argument("--local_ep", type=int, default=3, help="Local epochs")
    parser.add_argument('--lr_decay', type=float, default=0.95, help='Learning rate decay factor')
    parser.add_argument("--lr", type=float, default=0.001, help='Learning rate')
    parser.add_argument("--rl_lr", type=float, default=0.001, help='RL agent learning rate')
    parser.add_argument("--correction_rate", type=float, default=None, help='Fixed correction rate (alpha) for disconnected clients; if set, disables RL and uses this value')
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

    program = "PipeSFLV1 ResNet50 on CIFAR-10 with RL Correction"
    print(f"---------{program}----------")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('device:', device)

    # 初始化参数
    num_users = 4
    epochs = args.epochs
    disconnect_prob = args.disconnect_prob
    disconnect_round = args.disconnect_round
    local_ep = args.local_ep
    frac = 1
    lr = args.lr
    lr_decay = args.lr_decay
    train_times = []

    # 初始化模型
    net_glob_client = ResNet50_client_side().cpu()
    print(net_glob_client)

    net_glob_server = ResNet50_server_side(10).cpu()
    print(net_glob_server)

    # 初始化上一轮的全局模型参数
    prev_w_glob_client = {k: v.cpu() for k, v in copy.deepcopy(net_glob_client.state_dict()).items()}

    # 服务端损失和准确率收集
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

    # 客户端索引收集
    idx_collect = manager.list()
    idx_disconnected = manager.list()
    idx_round_disconnected = manager.list()
    idx_disconnected_time = manager.list([0] * num_users)

    l_epoch_check = False
    fed_check = False

    # 心跳监测
    heartbeat_queue = manager.Queue()
    monitor_process = multiprocessing.Process(target=monitor_heartbeats, args=(heartbeat_queue, num_users))
    monitor_process.start()

    # 初始化校正变量 (server corrections removed in global-server mode)
    client_corrections = {i: {k: torch.zeros_like(v) for k, v in net_glob_client.state_dict().items()} for i in
                          range(num_users)}

    # =============================================================================
    #                         数据预处理
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

    # 加载CIFAR-10数据集
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

    # 数据集划分: use cifar_user_dataset non-iid split similar to v2_cifar10.py
    dict_users = cifar_user_dataset(dataset_train, num_users, noniid_fraction=0.8)
    dict_users_test = dataset_iid(dataset_test, num_users)
    draw_data_distribution(dict_users, dataset_train, num_users,
                           save_path='output/data_distribution.png')

    # 如果指定了固定 correction_rate，则使用该值并禁用 RL（移除了 RLAgent）
    fixed_correction_rate = None
    if args.correction_rate is not None:
        fixed_correction_rate = float(args.correction_rate)
        print(f"Using fixed correction_rate = {fixed_correction_rate} (RL disabled)")
    else:
        print("No fixed correction_rate provided. RL module previously available has been removed for this experiment.")

    # 训练和测试
    net_glob_client.train()
    w_glob_client = net_glob_client.state_dict()

    for iter in range(epochs):
        idx_collect[:] = []
        idx_round_disconnected[:] = []
        start_time = time.time()
        m = max(int(frac * num_users), 1)
        idxs_users = np.random.choice(range(num_users), m, replace=False)
        w_locals_client = []

        global_seed = iter
        numpy.random.seed(global_seed)

        running.value = True

        if len(idx_disconnected) > 0 and len(idx_disconnected) < num_users:
            print(f"[sort idxs_users] sort idxs_users")
            idxs_users = sorted(idxs_users, key=lambda x: x not in idx_disconnected)
            print(f"[Round {iter}] Sorted idxs_users: {idxs_users}")

        for idx in idxs_users:
            print(f"[Round {iter}] Current user's idx: {idx}")
            if idx in idx_disconnected:
                local = Client(net_glob_client, idx, lr, net_glob_server, criterion, count1, idx_collect, num_users,
                               dataset_train=dataset_train,
                               dataset_test=dataset_test, idxs=dict_users[idx], idxs_test=dict_users_test[idx],
                               heartbeat_queue=heartbeat_queue, disconnect_prob=disconnect_prob,
                               idx_disconnected=idx_disconnected, running=running, is_disconnected=True,
                               idx_disconnected_time=idx_disconnected_time,
                               idx_round_disconnected=idx_round_disconnected, disconnect_seed=global_seed,
                               disconnect_round=disconnect_round, local_ep=local_ep)
                if idx not in idx_round_disconnected:
                    idx_round_disconnected.append(idx)
            else:
                local = Client(net_glob_client, idx, lr, net_glob_server, criterion, count1, idx_collect, num_users,
                               dataset_train=dataset_train,
                               dataset_test=dataset_test, idxs=dict_users[idx], idxs_test=dict_users_test[idx],
                               heartbeat_queue=heartbeat_queue, disconnect_prob=disconnect_prob,
                               idx_disconnected=idx_disconnected, running=running, is_disconnected=False,
                               idx_disconnected_time=idx_disconnected_time,
                               idx_round_disconnected=idx_round_disconnected, disconnect_seed=global_seed,
                               disconnect_round=disconnect_round, local_ep=local_ep)

            # 训练
            client_ret = local.train(net=copy.deepcopy(net_glob_client))
            if client_ret is None:
                w_client = None
            else:
                w_client = client_ret[0]

            if local.is_disconnected:
                prRed(f"Client{idx} 断开连接，使用RL动态校正变量")

                # 1. 获取当前状态
                offline_time = idx_disconnected_time[idx]
                if len(acc_test_collect) >= 1:
                    current_acc = acc_test_collect[-1]
                    prev_acc = acc_test_collect[-2] if len(acc_test_collect) >= 2 else current_acc
                else:
                    current_acc = 0.0
                    prev_acc = 0.0
                acc_change = current_acc - prev_acc

                if len(loss_test_collect) >= 1:
                    current_loss = loss_test_collect[-1]
                    prev_loss = loss_test_collect[-2] if len(loss_test_collect) >= 2 else current_loss
                else:
                    current_loss = 0.0
                    prev_loss = 0.0
                loss_change = prev_loss - current_loss
                data_diff = 0.5
                state = get_state(idx, offline_time, acc_change, loss_change, data_diff)

                # 使用固定修正率（如果提供），否则使用默认 0.5
                if fixed_correction_rate is not None:
                    correction_rate = fixed_correction_rate
                else:
                    correction_rate = 0.5
                print(f"[Decision] Client{idx} 修正率α = {correction_rate:.4f}")

                offline_w_client = {
                    k: (1 - correction_rate) * prev_w_glob_client[k] + correction_rate * (
                                prev_w_glob_client[k] - client_corrections[idx].get(k, torch.zeros_like(
                            prev_w_glob_client[k])))
                    for k in prev_w_glob_client.keys()
                }
                w_locals_client.append(offline_w_client)

                next_offline_time = offline_time + 1 if idx in idx_disconnected else 0
                next_acc = acc_test_collect[-1] if len(acc_test_collect) >= 1 else 0
                next_loss = loss_test_collect[-1] if len(loss_test_collect) >= 1 else 0
                prev_acc = acc_test_collect[-2] if len(acc_test_collect) >= 2 else 0
                prev_loss = loss_test_collect[-2] if len(loss_test_collect) >= 2 else 0

                next_state = get_state(idx, next_offline_time, next_acc - prev_acc, prev_loss - next_loss, data_diff)
                reward = get_reward(prev_acc, next_acc, prev_loss, next_loss)
                print(f"[RL Feedback] Client{idx} 奖励 = {reward:.4f}")

                # RL 已移除；不进行存储与学习，仅记录（保留接口点以便未来扩展）
            else:
                w_locals_client.append(w_client)

                global_update_client = net_glob_client.state_dict()
                for k in global_update_client.keys():
                    client_corrections[idx][k] = torch.clamp((global_update_client[k] - w_client[k]).float(), -1e3,
                                                             1e3)

                local.evaluate(w_client, ell=iter)

            cleanup_thread = threading.Thread(target=cleanup_client, args=(local,), daemon=True)
            cleanup_thread.start()

        running.value = False

        # 更新上一轮全局参数 (server updated in-place during training)
        prev_w_glob_client = {k: v.cpu() for k, v in copy.deepcopy(net_glob_client.state_dict()).items()}

        # 联邦平均 (client-side aggregation only)
        print("------------------------------------------------------------")
        print("------ Fed Server: Federation process at Client-Side -------")
        print("------------------------------------------------------------")

        if len(w_locals_client) > 0:
            w_glob_client = FedAvg(w_locals_client, client_corrections, model_type='client')
            net_glob_client.load_state_dict(w_glob_client)

        # 记录训练时间
        train_time = time.time() - start_time
        train_times.append(train_time)
        print("====================== PipeSFL V1 ========================")
        print('========== Train: Round {:3d} Time: {:2f}s ==============='.format(iter, train_time))
        print("==========================================================")

        # 更新离线倒计时
        for i in range(len(idx_disconnected_time)):
            if idx_disconnected_time[i] > 0:
                idx_disconnected_time[i] -= 1
                if idx_disconnected_time[i] == 0:
                    print(f"[Reconnect] Client{i} 将在下一轮重新连接")
                    idx_disconnected.remove(i)
                else:
                    print(f"[Reconnect] Client{i} 重新连接倒计时: {idx_disconnected_time[i]}")

        # 保护机制
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

        lr = lr * lr_decay

    # 训练完成后处理
    print("Training and Evaluation completed!")

    # 确保输出目录存在
    curve_dir = 'output/curve/cifar10/long_offline'
    model_dir = 'output/model/cifar10/long_offline'
    acc_dir = 'output/acc/cifar10/long_offline'
    loss_dir = 'output/loss/cifar10/long_offline'

    for directory in [curve_dir, model_dir, acc_dir, loss_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)

    # 绘制训练时间曲线
    plt.plot(range(epochs), train_times)
    plt.xlabel('Training Rounds')
    plt.ylabel('Training Time (s)')
    plt.title('Training Time Curve')
    plt.grid(True)
    prefix = f"_CIFAR10_constant_alpha_ep{args.epochs}_dp{args.disconnect_prob:.2f}_dr{args.disconnect_round}_le{args.local_ep}_lr{args.lr}"
    curve_filename = os.path.join(curve_dir, f'train_time_curve{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                          time.localtime()) + '.png')
    plt.savefig(curve_filename)
    plt.clf()

    # 保存模型
    client_model_filename = os.path.join(model_dir,
                                         f'Client{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.pth')
    server_model_filename = os.path.join(model_dir,
                                         f'Server{prefix}_' + time.strftime("%Y%m%d-%H%M%S", time.localtime()) + '.pth')

    torch.save(net_glob_client.state_dict(), client_model_filename)
    torch.save(net_glob_server.state_dict(), server_model_filename)
    print('Model saved successfully!')

    # 保存准确率和损失数据
    acc_train_collect_list = list(acc_train_collect)
    loss_train_collect_list = list(loss_train_collect)
    acc_test_collect_list = list(acc_test_collect)
    loss_test_collect_list = list(loss_test_collect)

    acc_train_df = pd.DataFrame(acc_train_collect_list)
    loss_train_df = pd.DataFrame(loss_train_collect_list)
    acc_test_df = pd.DataFrame(acc_test_collect_list)
    loss_test_df = pd.DataFrame(loss_test_collect_list)

    acc_train_filename = os.path.join(acc_dir, f'Client_Acc_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                           time.localtime()) + '.csv')
    acc_train_df.to_csv(acc_train_filename, index=False)

    loss_train_filename = os.path.join(loss_dir, f'Client_Loss_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                              time.localtime()) + '.csv')
    loss_train_df.to_csv(loss_train_filename, index=False)

    acc_test_filename = os.path.join(acc_dir, f'Server_Acc_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                          time.localtime()) + '.csv')
    acc_test_df.to_csv(acc_test_filename, index=False)

    loss_test_filename = os.path.join(loss_dir, f'Server_Loss_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                             time.localtime()) + '.csv')
    loss_test_df.to_csv(loss_test_filename, index=False)

    # 绘制准确率曲线
    plt.plot(range(epochs), acc_train_collect_list, label='Train Accuracy')
    plt.plot(range(epochs), acc_test_collect_list, label='Test Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Testing Accuracy')
    plt.legend()
    plt.grid(True)

    acc_curve_filename = os.path.join(curve_dir,
                                      f'acc_curve_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                 time.localtime()) + '.png')
    plt.savefig(acc_curve_filename)
    plt.clf()

    # 绘制损失曲线
    plt.plot(range(epochs), loss_train_collect_list, label='Train Loss')
    plt.plot(range(epochs), loss_test_collect_list, label='Test Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Testing Loss')
    plt.legend()
    plt.grid(True)

    loss_curve_filename = os.path.join(curve_dir,
                                       f'loss_curve_Corr{prefix}_' + time.strftime("%Y%m%d-%H%M%S",
                                                                                   time.localtime()) + '.png')
    plt.savefig(loss_curve_filename)
    print('Data saved successfully!')

    # 结束监测进程
    monitor_process.terminate()
    monitor_process.join()

    time.sleep(5)
    print("程序正常结束")