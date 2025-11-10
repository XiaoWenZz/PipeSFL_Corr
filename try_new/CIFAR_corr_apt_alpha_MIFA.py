# -*- coding: utf-8 -*-
"""
Full integrated Split Federated Learning (SFL) + MIFA (client-side) training script
==============================================================================

This file integrates your original pipeline (heartbeat, disconnections, dataset
splits, plotting/saving, multi-processing lists, etc.) and **replaces the old
correction/offline_w_client/FedAvg** logic with a **proper MIFA implementation**
(applied on the client-side model only), faithful to the paper:
  - Keep a per-client *accumulated gradient* memory Gi = (w_t - w_i) / lr
  - When a client is offline, reuse its *stale* Gi (do **not** synthesize weights)
  - At the end of each round: w_{t+1} = w_t - lr * mean_i(Gi)

Two MIFA modes are provided:
  1) per-client memory (stores all Gi)  [--mifa_memory_lite=False]
  2) memory-lite     (store only global G_bar) [--mifa_memory_lite=True]
Both are *mathematically equivalent*; memory-lite is much more memory-friendly.

NOTE:
- Client-side optimizer is set to SGD by default for theoretical alignment.
  (You can switch back to Adam via --client_optimizer)
- Server-side training loop is kept as in-place update on a single global server model
  (your SFL design). MIFA is applied **only** to the client-side weights.
- Heartbeat/disconnection simulation is preserved. When a client disconnects
  during its local training, its current round Gi is simply not updated; the
  previous Gi remains in memory, per MIFA.

Run example:
  python sfl_mifa_full.py \
    --epochs 50 --disconnect_prob 0.4 --disconnect_round 3 \
    --local_ep 3 --lr 0.001 --lr_decay 1.0 --noniid_fraction 1.0 \
    --alpha 0.2 --balanced_split False --mifa_memory_lite True

"""
from __future__ import annotations
import os
import math
import time
import copy
import argparse
import random
import threading
import multiprocessing
import queue
from collections import defaultdict

import numpy as np
import numpy.random
import pandas as pd
from pandas import DataFrame

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --------------------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------------------
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = os.environ.get("CUDA_VISIBLE_DEVICES", "0")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --------------------------------------------------------------------------------------
# Pretty print helpers
# --------------------------------------------------------------------------------------
def prRed(s):
    print("[91m" + str(s) + "[00m")

def prGreen(s):
    print("[92m" + str(s) + "[00m")

# --------------------------------------------------------------------------------------
# Models (Split ResNet50-like tiny client + larger server)
# --------------------------------------------------------------------------------------
class BasicBlock(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes*BasicBlock.expansion, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(planes*BasicBlock.expansion)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes * BasicBlock.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes*BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes*BasicBlock.expansion),
            )
    def forward(self, x):
        x1 = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x2 = F.leaky_relu(self.bn2(self.conv2(x1)), inplace=True)
        x3 = self.bn3(self.conv3(x2))
        out = F.leaky_relu(x3 + self.shortcut(x), inplace=True)
        return out

class ResNet50_client_side(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1   = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64, 3, stride=1)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        return x

class ResNet50_server_side(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.in_planes = 256
        self.layer2 = self._make_layer(BasicBlock, 128, 4, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 6, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for st in strides:
            layers.append(block(self.in_planes, planes, st))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --------------------------------------------------------------------------------------
# SFL Server: train/eval on server-side model given client activations
# --------------------------------------------------------------------------------------
acc_train_collect = []
loss_train_collect = []
acc_test_collect  = []
loss_test_collect = []

batch_acc_test = []
batch_loss_test = []

acc_avg_all_user_train_global = 0.0
loss_avg_all_user_train_global = 0.0

l_epoch_check = False
fed_check = False

criterion = nn.CrossEntropyLoss()
net_glob_server = None  # will be created in main


def calculate_accuracy(logits, y):
    preds = logits.argmax(dim=1, keepdim=True)
    correct = preds.eq(y.view_as(preds)).sum()
    return 100.0 * correct.float() / preds.shape[0]


def train_server(fx_client, y, l_epoch_count, l_epoch, idx, len_batch, lr, 
                 batch_acc_train, batch_loss_train, count1,
                 loss_train_collect_user, acc_train_collect_user,
                 idx_collect, idx_disconnected, idx_round_disconnected,
                 num_users):
    """One step server-side update given client activations.
    Keeps a single global server model (in-place update).
    Returns gradient wrt client activations for split backprop.
    """
    global l_epoch_check, fed_check, net_glob_server
    net_glob_server = net_glob_server.to(device)
    net_glob_server.train()

    optimizer_server = torch.optim.Adam(net_glob_server.parameters(), lr=lr)

    optimizer_server.zero_grad()
    fx_client = fx_client.requires_grad_(True)

    fx_server = net_glob_server(fx_client)
    loss = criterion(fx_server, y)
    acc = calculate_accuracy(fx_server, y)

    loss.backward()
    if torch.isnan(fx_client.grad).any():
        prRed(f"[Error] Server grad NaN, mark Client {idx} as disconnected this round")
        idx_round_disconnected.append(idx)
        return None

    dfx_client = fx_client.grad.clone().detach()
    optimizer_server.step()

    batch_loss_train.append(loss.item())
    batch_acc_train.append(acc.item())

    count1 += 1
    if count1 == len_batch * l_epoch:
        acc_avg_train = sum(batch_acc_train) / max(1, len(batch_acc_train))
        loss_avg_train = sum(batch_loss_train) / max(1, len(batch_loss_train))
        batch_acc_train.clear(); batch_loss_train.clear(); count1 = 0
        prRed(f'Client{idx} Train => Local Epoch: {l_epoch_count} 	Acc: {acc_avg_train:.3f} 	Loss: {loss_avg_train:.4f}')

        if l_epoch_count == l_epoch - 1:
            l_epoch_check = True
            loss_train_collect_user.append(loss_avg_train)
            acc_train_collect_user.append(acc_avg_train)
            if idx not in idx_collect:
                if idx in idx_disconnected:
                    prRed(f"[Warn] Client{idx} in idx_disconnected but collected; race condition?")
                else:
                    idx_collect.append(idx)

        if len(idx_collect) + len(idx_round_disconnected) == num_users:
            fed_check = True
            if len(acc_train_collect_user) > 0:
                acc_avg_all_user_train = sum(acc_train_collect_user) / len(acc_train_collect_user)
                loss_avg_all_user_train = sum(loss_train_collect_user) / len(loss_train_collect_user)
            else:
                if len(acc_train_collect) == 0:
                    acc_avg_all_user_train = 0.0
                    loss_avg_all_user_train = 0.0
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
    global net_glob_server, criterion
    global batch_acc_test, batch_loss_test
    global loss_test_collect, acc_test_collect
    global count2, num_users, l_epoch_check, fed_check
    global loss_test_collect_user, acc_test_collect_user
    global acc_avg_all_user_train_global, loss_avg_all_user_train_global

    net_glob_server = net_glob_server.to(device)
    net_glob_server.eval()

    with torch.no_grad():
        fx_client = fx_client.to(device)
        y = y.to(device)
        fx_server = net_glob_server(fx_client)
        loss = criterion(fx_server, y)
        acc  = calculate_accuracy(fx_server, y)

        batch_loss_test.append(loss.item())
        batch_acc_test.append(acc.item())

        count2 += 1
        if count2 == len_batch:
            acc_avg_test = sum(batch_acc_test) / max(1, len(batch_acc_test))
            loss_avg_test = sum(batch_loss_test) / max(1, len(batch_loss_test))
            batch_acc_test.clear(); batch_loss_test.clear(); count2 = 0
            print(f'Client{idx} Test => Acc: {acc_avg_test:.3f} 	Loss: {loss_avg_test:.4f}')

            if l_epoch_check:
                l_epoch_check = False
                loss_test_collect_user.append(loss_avg_test)
                acc_test_collect_user.append(acc_avg_test)

            if fed_check:
                fed_check = False
                acc_avg_all_user = sum(acc_test_collect_user) / max(1, len(acc_test_collect_user)) if len(acc_test_collect_user) else 0.0
                loss_avg_all_user = sum(loss_test_collect_user) / max(1, len(loss_test_collect_user)) if len(loss_test_collect_user) else 0.0
                loss_test_collect.append(loss_avg_all_user)
                acc_test_collect.append(acc_avg_all_user)
                acc_test_collect_user.clear(); loss_test_collect_user.clear()
                print("====================== SERVER V1 ==========================")
                print(f' Train: Round {ell:3d}, Avg Acc {acc_avg_all_user_train_global:.3f} | Avg Loss {loss_avg_all_user_train_global:.3f}')
                print(f' Test : Round {ell:3d}, Avg Acc {acc_avg_all_user:.3f} | Avg Loss {loss_avg_all_user:.3f}')
                print("==========================================================")

# --------------------------------------------------------------------------------------
# Dataset split helpers
# --------------------------------------------------------------------------------------
class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)
    def __len__(self):
        return len(self.idxs)
    def __getitem__(self, item):
        x, y = self.dataset[self.idxs[item]]
        return x, y


def dataset_iid(dataset, num_users):
    labels = [label for _, label in dataset]
    class_idxs = defaultdict(list)
    for idx, lab in enumerate(labels):
        class_idxs[lab].append(idx)
    dict_users = {}
    per_user = len(dataset) // num_users
    all_idx = np.random.permutation(len(dataset)).tolist()
    for u in range(num_users):
        pick = all_idx[u*per_user:(u+1)*per_user]
        dict_users[u] = set(pick)
    return dict_users


def cifar_user_dataset_dirichlet(dataset, num_users, noniid_fraction, alpha=0.1, balanced=True, seed=None):
    if seed is not None:
        np.random.seed(seed)
    total_items = len(dataset)
    num_noniid_items = int(total_items * noniid_fraction)
    num_iid_items = total_items - num_noniid_items
    all_indices = np.arange(total_items)

    dict_users = [set() for _ in range(num_users)]

    # IID portion
    if num_iid_items > 0:
        iid_indices = np.random.choice(all_indices, num_iid_items, replace=False)
        all_indices = np.setdiff1d(all_indices, iid_indices)
        per_user_iid = num_iid_items // num_users
        for i in range(num_users):
            chosen = iid_indices[i*per_user_iid:(i+1)*per_user_iid]
            dict_users[i].update(chosen)

    # non-IID portion via Dirichlet
    if num_noniid_items > 0:
        noniid_indices = all_indices
        labels = np.array([dataset[i][1] for i in noniid_indices])
        classes = np.unique(labels)
        class_indices = {c: np.where(labels == c)[0] for c in classes}
        class_props = np.random.dirichlet([alpha] * num_users, size=len(classes))
        user_data = defaultdict(list)
        for c_idx, c in enumerate(classes):
            idxs = class_indices[c]
            np.random.shuffle(idxs)
            props = class_props[c_idx]
            cuts = (np.cumsum(props) * len(idxs)).astype(int)
            parts = np.split(idxs, cuts[:-1])
            for u, part in enumerate(parts):
                user_data[u].extend(noniid_indices[part])
        for u in range(num_users):
            dict_users[u].update(user_data[u])

    if balanced:
        total_per_user = total_items // num_users
        all_used = set(); extra_pool = []
        for u in range(num_users):
            data_u = list(dict_users[u])
            if len(data_u) > total_per_user:
                keep = np.random.choice(data_u, total_per_user, replace=False)
                extra = list(set(data_u) - set(keep))
                dict_users[u] = set(keep)
                extra_pool.extend(extra)
            all_used.update(dict_users[u])
        remaining = list(set(np.arange(total_items)) - all_used)
        remaining.extend(extra_pool)
        np.random.shuffle(remaining)
        ptr = 0
        for u in range(num_users):
            need = total_per_user - len(dict_users[u])
            if need > 0:
                add = remaining[ptr:ptr+need]
                dict_users[u].update(add)
                ptr += need
    return {i: set(dict_users[i]) for i in range(num_users)}

# --------------------------------------------------------------------------------------
# Heartbeat / client class
# --------------------------------------------------------------------------------------
class Client(object):
    def __init__(self, net_client_model, idx, lr, net_glob_server, criterion, idx_collect, num_users,
                 dataset_train=None, dataset_test=None, idxs=None, idxs_test=None, heartbeat_queue=None,
                 disconnect_prob=0.001, idx_disconnected=None, is_disconnected=False, idx_disconnected_time=None,
                 idx_round_disconnected=None, disconnect_seed=0, disconnect_round=1, local_ep=1,
                 client_optimizer='sgd'):
        self.disconnect_prob = disconnect_prob
        self.is_disconnected = is_disconnected
        self.heartbeat_queue = heartbeat_queue
        self.idx = idx
        self.lr = lr
        self.local_ep = local_ep
        self.client_optimizer = client_optimizer
        self.net_glob_server = net_glob_server
        self.criterion = criterion
        self.batch_acc_train = []
        self.batch_loss_train = []
        self.loss_train_collect_user = []
        self.acc_train_collect_user = []
        self.idx_collect = idx_collect
        self.num_users = num_users
        self.ldr_train = DataLoader(DatasetSplit(dataset_train, idxs), batch_size=1024, shuffle=True)
        self.ldr_test  = DataLoader(DatasetSplit(dataset_test,  idxs_test), batch_size=1024, shuffle=True)
        self.disconnect_seed = disconnect_seed
        self.rng = numpy.random.default_rng(seed=self.disconnect_seed + self.idx)
        self.disconnect_round = disconnect_round
        self.idx_disconnected = idx_disconnected
        self.idx_round_disconnected = idx_round_disconnected
        self.idx_disconnected_time = idx_disconnected_time
        self.status = "idle"
        self.heartbeat_interval = 5
        self.stop_heartbeat_flag = False
        self.heartbeat_thread = threading.Thread(target=self.send_heartbeat, daemon=True)
        self.heartbeat_thread.start()

    def send_heartbeat(self):
        try:
            while not self.stop_heartbeat_flag:
                if not self.is_disconnected:
                    rnd = self.rng.random()
                    self.rng = numpy.random.default_rng(seed=self.disconnect_seed + self.idx)
                    print(f"[send_heartbeat] Client{self.idx} 随机数: {rnd}")
                    self.is_disconnected = (rnd < self.disconnect_prob and self.status in ["training", "testing"]) 
                    if self.is_disconnected:
                        print(f"[Disconnect] Client{self.idx} 断开 (p={self.disconnect_prob*100:.1f}%)")
                        self.heartbeat_queue.put((self.idx, "disconnected", time.strftime("%Y-%m-%d %H:%M:%S")))
                        if self.idx not in self.idx_disconnected:
                            if self.idx in self.idx_collect:
                                prRed(f"[Warn] Client{self.idx} in idx_collect, not adding to idx_disconnected")
                            else:
                                self.idx_disconnected.append(self.idx)
                                self.idx_round_disconnected.append(self.idx)
                        self.idx_disconnected_time[self.idx] = self.disconnect_round
                        time.sleep(self.heartbeat_interval)
                        continue
                if self.is_disconnected:
                    time.sleep(self.heartbeat_interval)
                    continue
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
                self.heartbeat_queue.put((self.idx, self.status, timestamp))
                time.sleep(self.heartbeat_interval)
        except (EOFError, BrokenPipeError):
            print(f"[Info] Client{self.idx} 心跳线程退出")
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
                    acc_avg_all_user_train = 0.0
                    loss_avg_all_user_train = 0.0
                else:
                    acc_avg_all_user_train = acc_train_collect[-1]
                    loss_avg_all_user_train = loss_train_collect[-1]
            global acc_avg_all_user_train_global, loss_avg_all_user_train_global
            acc_avg_all_user_train_global = acc_avg_all_user_train
            loss_avg_all_user_train_global = loss_avg_all_user_train
            acc_train_collect.append(acc_avg_all_user_train)
            if len(acc_test_collect) == 0:
                acc_test_collect.append(0.0); loss_test_collect.append(0.0)
            else:
                acc_test_collect.append(acc_test_collect[-1])
                loss_test_collect.append(loss_test_collect[-1])
        return None, None

    def train(self, net: nn.Module):
        global l_epoch_check, fed_check
        if self.is_disconnected:
            print(f"[Skip] Client{self.idx} 断开，跳过训练")
            if self.idx not in self.idx_disconnected:
                if self.idx in self.idx_collect:
                    prRed(f"[Warn] Client{self.idx} in idx_collect")
                else:
                    self.idx_disconnected.append(self.idx)
            if (not fed_check) and len(self.idx_collect) + len(self.idx_disconnected) == self.num_users:
                self.update_fed_check()
            return None, None

        try:
            self.status = "training"
            net = net.to(device)
            self.net_glob_server = self.net_glob_server.to(device)
            net.train()
            if self.client_optimizer.lower() == 'adam':
                optimizer_client = torch.optim.Adam(net.parameters(), lr=self.lr)
            else:
                optimizer_client = torch.optim.SGD(net.parameters(), lr=self.lr, momentum=0.0)

            for ep in range(self.local_ep):
                if self.is_disconnected:
                    prRed(f"[Abort] Client{self.idx} 训练中断开")
                    if self.idx not in self.idx_disconnected:
                        if self.idx in self.idx_collect:
                            prRed(f"[Warn] Client{self.idx} 在 idx_collect 中")
                        else:
                            self.idx_disconnected.append(self.idx)
                    if (not fed_check) and len(self.idx_collect) + len(self.idx_disconnected) == self.num_users:
                        self.update_fed_check()
                    break
                len_batch = len(self.ldr_train)

                count1 = 0
                for batch_idx, (images, labels) in enumerate(self.ldr_train):
                    if self.is_disconnected:
                        prRed(f"[Abort] Client{self.idx} 训练中断开")
                        if self.idx not in self.idx_disconnected:
                            if self.idx in self.idx_collect:
                                prRed(f"[Warn] Client{self.idx} 在 idx_collect 中")
                            else:
                                self.idx_disconnected.append(self.idx)
                        if (not fed_check) and len(self.idx_collect) + len(self.idx_disconnected) == self.num_users:
                            self.update_fed_check()
                        break

                    images, labels = images.to(device), labels.to(device)
                    optimizer_client.zero_grad()
                    fx = net(images)
                    client_fx = fx.clone().detach().to(device)
                    count1 += 1

                    dfx = train_server(client_fx, labels, ep, self.local_ep, self.idx,
                                       len_batch, self.lr,
                                       self.batch_acc_train, self.batch_loss_train, count1,
                                       self.loss_train_collect_user, self.acc_train_collect_user,
                                       self.idx_collect, self.idx_disconnected, self.idx_round_disconnected,
                                       self.num_users)
                    fx.backward(dfx)
                    for p in net.parameters():
                        assert p.grad is None or p.grad.dtype == torch.float32
                    optimizer_client.step()

            net.to('cpu'); self.net_glob_server.to('cpu')
            for p in net.parameters():
                if torch.isnan(p).any():
                    prRed(f"[Error] Client{self.idx} 参数包含 NaN")
            return net.cpu().state_dict(),
        finally:
            self.status = "idle"

    def evaluate(self, w_client, ell):
        if self.is_disconnected:
            print(f"[Skip] Client{self.idx} 断开，跳过测试")
            return
        try:
            self.status = "testing"
            net = ResNet50_client_side().cpu()
            net.load_state_dict(w_client)
            net = net.to(device)
            net.eval()

            for p in net.parameters():
                if torch.isnan(p).any():
                    prRed(f"[Error] Client{self.idx} 模型参数NaN，跳测")
                    return
            with torch.no_grad():
                for images, labels in self.ldr_test:
                    if torch.isnan(images).any() or torch.isnan(labels).any():
                        prRed(f"[Error] Client{self.idx} 测试数据含NaN")
                        return
                len_batch = len(self.ldr_test)
                for images, labels in self.ldr_test:
                    if self.is_disconnected:
                        prRed(f"[Abort] Client{self.idx} 测试中断开")
                        break
                    images, labels = images.to(device), labels.to(device)
                    fx = net(images)
                    evaluate_server(fx, labels, self.idx, len_batch, ell)
        finally:
            self.status = "idle"; net.to('cpu')

# --------------------------------------------------------------------------------------
# MIFA helpers (state_dict math)
# --------------------------------------------------------------------------------------
def sd_copy(sd):
    return {k: v.clone().detach().cpu() for k, v in sd.items()}

def sd_like(sd, fill=0.0):
    return {k: torch.zeros_like(v).fill_(fill) for k, v in sd.items()}

def sd_add(a, b, alpha=1.0):
    return {k: a[k] + alpha * b[k] for k in a.keys()}

def sd_sub(a, b):
    return {k: a[k] - b[k] for k in a.keys()}

def sd_mul(a, scalar):
    return {k: a[k] * scalar for k in a.keys()}

def sd_div(a, scalar):
    return {k: a[k] / scalar for k in a.keys()}

# --------------------------------------------------------------------------------------
# Heartbeat monitor process
# --------------------------------------------------------------------------------------
def monitor_heartbeats(heartbeat_queue, num_users):
    client_status = {i: {"status": "idle", "last_heartbeat": "", "type": "normal"} for i in range(num_users)}
    while True:
        try:
            idx, status, timestamp = heartbeat_queue.get(timeout=30)
            client_status[idx] = {"status": status, "last_heartbeat": timestamp,
                                  "type": "normal" if status != "disconnected" else "disconnected"}
        except queue.Empty:
            for idx in range(num_users):
                if client_status[idx]["type"] != "disconnected" and client_status[idx]["status"] in ["training","testing"]:
                    prRed(f"[Error] Client {idx} exited unexpectedly!")
                    client_status[idx] = {"status": "idle", "last_heartbeat": "", "type": "disconnected"}
        except IOError as e:
            if "[WinError 232]" in str(e):
                print("[Info] 管道正常关闭，退出监测...")
                return
        except Exception as e:
            prRed(f"[Error] heartbeat monitor: {e}")

# --------------------------------------------------------------------------------------
# Visualization helper
# --------------------------------------------------------------------------------------
def draw_data_distribution(dict_users, dataset, num_users, save_path='data_distribution.png'):
    client_dist = {i: [0] * 10 for i in range(num_users)}
    for client_idx, indices in dict_users.items():
        labels = [dataset[idx][1] for idx in indices]
        for label in labels:
            client_dist[client_idx][label] += 1
    fig, axes = plt.subplots(nrows=num_users, ncols=1, figsize=(12, 3 * num_users))
    if num_users == 1:
        axes = [axes]
    for i in range(num_users):
        ax = axes[i]
        ax.bar(range(10), client_dist[i])
        ax.set_title(f'Client {i} Data Distribution')
        ax.set_xlabel('Class'); ax.set_ylabel('Count')
        ax.set_xticks(range(10)); ax.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout(); plt.savefig(save_path); plt.close()

# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
if __name__ == '__main__':
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        print(torch.cuda.get_device_name(0))

    parser = argparse.ArgumentParser(description='SFL + MIFA (client-side) full integration')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--disconnect_prob', type=float, default=0.40)
    parser.add_argument('--disconnect_round', type=int, default=3)
    parser.add_argument('--local_ep', type=int, default=3)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--lr_decay', type=float, default=1.0)
    parser.add_argument('--noniid_fraction', type=float, default=1.0)
    parser.add_argument('--alpha', type=float, default=0.2, help='Dirichlet concentration')
    parser.add_argument('--balanced_split', type=lambda x: str(x).lower()=='true', default=False)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--client_optimizer', type=str, default='sgd', choices=['sgd','adam'])
    parser.add_argument('--mifa_memory_lite', type=lambda x: str(x).lower()=='true', default=True,
                        help='If True, store only global G_bar; else store per-client Gi')
    args = parser.parse_args()

    # Seeds
    random.seed(args.seed); np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    print(f"Available GPUs: {[i for i in range(torch.cuda.device_count())]}")
    print(f"device: {device}")

    # Init models
    net_glob_client = ResNet50_client_side().cpu()
    net_glob_server = ResNet50_server_side(10).cpu()
    print(net_glob_client); print(net_glob_server)

    prev_w_glob_client = sd_copy(net_glob_client.state_dict())

    manager = multiprocessing.Manager()
    running = manager.Value('b', True)

    # Server metrics collection (shared lists)
    loss_train_collect_m = manager.list()
    acc_train_collect_m  = manager.list()
    loss_test_collect_m  = manager.list()
    acc_test_collect_m   = manager.list()

    # Tie globals to manager-backed lists (for minimal changes in server funcs)
    loss_train_collect[:] = []
    acc_train_collect[:]  = []
    loss_test_collect[:]  = []
    acc_test_collect[:]   = []

    loss_test_collect_user = manager.list()
    acc_test_collect_user  = manager.list()

    idx_collect          = manager.list()
    idx_disconnected     = manager.list()
    idx_round_disconnected = manager.list()
    num_users = 4
    idx_disconnected_time = manager.list([0]*num_users)

    # Heartbeat monitor
    heartbeat_queue = manager.Queue()
    monitor_process = multiprocessing.Process(target=monitor_heartbeats, args=(heartbeat_queue, num_users))
    monitor_process.start()

    # Dataset & split
    mean = [0.4914, 0.4822, 0.4465]; std = [0.2023, 0.1994, 0.2010]
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

    dataset_train = datasets.CIFAR10(root='./data/cifar10', train=True, download=True, transform=train_transforms)
    dataset_test  = datasets.CIFAR10(root='./data/cifar10', train=False, download=True, transform=test_transforms)

    dict_users = cifar_user_dataset_dirichlet(dataset_train, num_users,
                    noniid_fraction=args.noniid_fraction, alpha=args.alpha,
                    balanced=args.balanced_split, seed=args.seed)
    dict_users_test = dataset_iid(dataset_test, num_users)

    os.makedirs('output', exist_ok=True)
    os.makedirs('output/curve/cifar10/long_offline', exist_ok=True)
    os.makedirs('output/model/cifar10/long_offline', exist_ok=True)
    os.makedirs('output/acc/cifar10/long_offline', exist_ok=True)
    os.makedirs('output/loss/cifar10/long_offline', exist_ok=True)

    draw_data_distribution(dict_users, dataset_train, num_users, save_path='output/data_distribution.png')

    # Print sample counts
    try:
        client_sample_counts = [len(dict_users[i]) for i in range(num_users)]
        total_samples = sum(client_sample_counts)
        print(f"[Info] Client sample counts: {client_sample_counts}")
        print(f"[Info] Total samples: {total_samples}")
    except Exception as e:
        prRed(f"[Warn] sample counts failed: {e}")

    # ---------------------- MIFA memory init ----------------------
    if args.mifa_memory_lite:
        # memory-lite: store only global G_bar and per-client last Gi for incremental update
        g_last = {i: sd_like(net_glob_client.state_dict(), 0.0) for i in range(num_users)}
        G_bar  = sd_like(net_glob_client.state_dict(), 0.0)
    else:
        # full: store Gi for each client
        g_memory = {i: sd_like(net_glob_client.state_dict(), 0.0) for i in range(num_users)}
    # ---------------------------------------------------------------

    count2 = 0
    train_times = []

    for it in range(args.epochs):
        idx_collect[:] = []
        idx_round_disconnected[:] = []
        start_time = time.time()

        # snapshot w_t for this round
        w_t_client = sd_copy(net_glob_client.state_dict())

        m = max(1, num_users)  # you activated all users each round in original; keep it.
        idxs_users = np.random.choice(range(num_users), m, replace=False)

        if len(idx_disconnected) > 0 and len(idx_disconnected) < num_users:
            idxs_users = sorted(idxs_users, key=lambda x: x not in idx_disconnected)
            print(f"[Round {it}] Sorted idxs_users: {idxs_users}")

        for idx in idxs_users:
            print(f"[Round {it}] Current user's idx: {idx}")
            if idx in idx_disconnected:
                local = Client(net_glob_client, idx, args.lr, net_glob_server, criterion, idx_collect, num_users,
                               dataset_train=dataset_train, dataset_test=dataset_test,
                               idxs=dict_users[idx], idxs_test=dict_users_test[idx],
                               heartbeat_queue=heartbeat_queue, disconnect_prob=args.disconnect_prob,
                               idx_disconnected=idx_disconnected, is_disconnected=True,
                               idx_disconnected_time=idx_disconnected_time, idx_round_disconnected=idx_round_disconnected,
                               disconnect_seed=it, disconnect_round=args.disconnect_round, local_ep=args.local_ep,
                               client_optimizer=args.client_optimizer)
                if idx not in idx_round_disconnected:
                    idx_round_disconnected.append(idx)
            else:
                local = Client(net_glob_client, idx, args.lr, net_glob_server, criterion, idx_collect, num_users,
                               dataset_train=dataset_train, dataset_test=dataset_test,
                               idxs=dict_users[idx], idxs_test=dict_users_test[idx],
                               heartbeat_queue=heartbeat_queue, disconnect_prob=args.disconnect_prob,
                               idx_disconnected=idx_disconnected, is_disconnected=False,
                               idx_disconnected_time=idx_disconnected_time, idx_round_disconnected=idx_round_disconnected,
                               disconnect_seed=it, disconnect_round=args.disconnect_round, local_ep=args.local_ep,
                               client_optimizer=args.client_optimizer)

            # client-side training
            client_ret = local.train(net=copy.deepcopy(net_glob_client))
            if client_ret is None:
                w_client = None
            else:
                w_client = client_ret[0]

            if local.is_disconnected:
                prRed(f"Client{idx} 断开，本轮不更新其 Gi（沿用上次记忆）")
                # do nothing for MIFA; stale Gi will be used
            else:
                # compute Gi = (w_t - w_i) / lr
                delta = sd_sub(w_t_client, sd_copy(w_client))
                Gi = sd_div(delta, args.lr)
                if args.mifa_memory_lite:
                    # incremental update: G_bar += (Gi - g_last[idx]) / N
                    inc = sd_sub(Gi, g_last[idx])
                    G_bar = sd_add(G_bar, sd_mul(inc, 1.0/num_users))
                    g_last[idx] = Gi
                else:
                    # full memory: store per-client Gi
                    g_memory[idx] = Gi

                # optional eval on this client snapshot
                local.evaluate(w_client, ell=it)

            # async cleanup
            threading.Thread(target=lambda c: c.stop_heartbeat(), args=(local,), daemon=True).start()

        # End of round: global update via MIFA
        if args.mifa_memory_lite:
            # w_{t+1} = w_t - lr * G_bar
            w_next = sd_add(w_t_client, sd_mul(G_bar, -args.lr))
        else:
            # G_avg = mean_i Gi
            G_avg = sd_like(w_t_client, 0.0)
            for i in range(num_users):
                G_avg = sd_add(G_avg, g_memory[i], alpha=1.0/num_users)
            w_next = sd_add(w_t_client, sd_mul(G_avg, -args.lr))

        net_glob_client.load_state_dict(w_next)
        prev_w_glob_client = sd_copy(w_next)

        # training time record
        train_time = time.time() - start_time
        train_times.append(train_time)
        print("====================== SFL + MIFA =======================")
        print(f" Round {it:3d} | Time: {train_time:.2f}s")
        print("========================================================")

        # Re-connection countdown
        for i in range(len(idx_disconnected_time)):
            if idx_disconnected_time[i] > 0:
                idx_disconnected_time[i] -= 1
                if idx_disconnected_time[i] == 0:
                    print(f"[Reconnect] Client{i} 将在下一轮重新连接")
                    if i in idx_disconnected:
                        idx_disconnected.remove(i)
                else:
                    print(f"[Reconnect] Client{i} 倒计时: {idx_disconnected_time[i]}")

        # Book-keeping safety
        if len(acc_train_collect) > 0:
            if len(acc_train_collect) < it + 1:
                acc_train_collect.append(acc_train_collect[-1])
                loss_train_collect.append(loss_train_collect[-1])
        if len(acc_test_collect) > 0:
            if len(acc_test_collect) < len(acc_train_collect):
                acc_test_collect.append(acc_test_collect[-1])
                loss_test_collect.append(loss_test_collect[-1])

        # LR decay (if any)
        if args.lr_decay != 1.0:
            for g in net_glob_client.state_dict().values():
                pass  # client LR is provided via args each round; keep simple here

    # ------------------------- Post: save outputs -------------------------
    print("Training and Evaluation completed!")

    curve_dir = 'output/curve/cifar10/long_offline'
    model_dir = 'output/model/cifar10/long_offline'
    acc_dir   = 'output/acc/cifar10/long_offline'
    loss_dir  = 'output/loss/cifar10/long_offline'

    # Save models
    ts = time.strftime("%Y%m%d-%H%M%S", time.localtime())
    client_model_filename = os.path.join(model_dir, f'Client_MIFA_{ts}.pth')
    server_model_filename = os.path.join(model_dir, f'Server_MIFA_{ts}.pth')
    torch.save(net_glob_client.state_dict(), client_model_filename)
    torch.save(net_glob_server.state_dict(), server_model_filename)
    print('Model saved successfully!')

    # Save curves arrays
    acc_train_df = pd.DataFrame(acc_train_collect)
    loss_train_df = pd.DataFrame(loss_train_collect)
    acc_test_df  = pd.DataFrame(acc_test_collect)
    loss_test_df = pd.DataFrame(loss_test_collect)

    acc_train_df.to_csv(os.path.join(acc_dir, f'Client_Acc_MIFA_{ts}.csv'), index=False)
    loss_train_df.to_csv(os.path.join(loss_dir, f'Client_Loss_MIFA_{ts}.csv'), index=False)
    acc_test_df.to_csv(os.path.join(acc_dir, f'Server_Acc_MIFA_{ts}.csv'), index=False)
    loss_test_df.to_csv(os.path.join(loss_dir, f'Server_Loss_MIFA_{ts}.csv'), index=False)

    # Plot: time per round
    plt.plot(range(len(train_times)), train_times)
    plt.xlabel('Rounds'); plt.ylabel('Time (s)'); plt.title('Training Time per Round'); plt.grid(True)
    plt.savefig(os.path.join(curve_dir, f'train_time_curve_MIFA_{ts}.png')); plt.clf()

    # Plot: acc/loss (if collected)
    if len(acc_train_collect) > 0:
        plt.plot(range(len(acc_train_collect)), acc_train_collect, label='Train Acc')
        if len(acc_test_collect) > 0:
            plt.plot(range(len(acc_test_collect)), acc_test_collect, label='Test Acc')
        plt.legend(); plt.grid(True); plt.title('Accuracy'); plt.xlabel('Rounds'); plt.ylabel('Accuracy')
        plt.savefig(os.path.join(curve_dir, f'acc_curve_MIFA_{ts}.png')); plt.clf()
    if len(loss_train_collect) > 0:
        plt.plot(range(len(loss_train_collect)), loss_train_collect, label='Train Loss')
        if len(loss_test_collect) > 0:
            plt.plot(range(len(loss_test_collect)), loss_test_collect, label='Test Loss')
        plt.legend(); plt.grid(True); plt.title('Loss'); plt.xlabel('Rounds'); plt.ylabel('Loss')
        plt.savefig(os.path.join(curve_dir, f'loss_curve_MIFA_{ts}.png')); plt.clf()

    # stop heartbeat monitor
    try:
        monitor_process.terminate(); monitor_process.join(timeout=5)
    except Exception:
        pass

    time.sleep(1)
    print("程序正常结束")
