#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 15:37
# @Author  : Fun.
# @File    : train.py
# @Software: PyCharm

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from dataset import MyDataset
from dataloder import MyDataLoader
from models import Transformer

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 创建训练数据加载器
train_dataloader = MyDataLoader(csv_file='dataset/THUNews/5_5000/train.csv', batch_size=4, max_length=256)\
    .get_data_loader()

# 创建验证数据加载器
valid_dataset = MyDataset(csv_file='dataset/THUNews/5_5000/dev.csv')
valid_dataloader = MyDataLoader(csv_file='dataset/THUNews/5_5000/dev.csv', batch_size=4, max_length=256)\
    .get_data_loader()

# 创建测试数据加载器
# test_dataset = MyDataset(csv_file='dataset/THUNews/5_5000/test.csv')
# test_dataloader = MyDataLoader(csv_file='dataset/THUNews/5_5000/test.csv', batch_size=4, max_length=256)\
#     .get_data_loader()


# dataset = MyDataset(csv_file='mydata/train.csv')
# train_dataloader = MyDataLoader(csv_file='mydata/train.csv', batch_size=4, max_length=256)\
#     .get_data_loader()
# valid_dataloader = MyDataLoader(csv_file='mydata/test.csv', batch_size=4, max_length=256)\
#     .get_data_loader()
# test_dataloader = MyDataLoader(csv_file='mydata/test.csv', batch_size=4, max_length=256)\
#     .get_data_loader()

# # 打印前几个样本
# print("Printing samples from the dataset:")
# for i in range(min(5, len(dataset))):  # 打印前5个样本
#     sample = dataset[i]
#     print(f"Sample {i}:")
#     print("Text:", sample['text'])
#     print("Label:", sample['label'])
#     print()
#
# # 打印一个批次的数据
# print("Printing a batch of data from the data loader:")
# batch = next(iter(data_loader))
# inputs, labels = batch
# print("Inputs:")
# print(inputs)
# print("Labels:")
# print(labels)

# 设置一些超参数（d_model能被n_heads整除）
input_size = 21128
output_size = 5
d_model = 128
n_heads = 8
d_ff = 2048
n_layers = 6
model = Transformer(input_size=input_size, output_size=output_size, d_model=d_model, n_heads=n_heads,
                    d_ff=d_ff, n_layers=n_layers).to(device)


def train_loop(dataloader, model, loss_fn, optimizer, lr_scheduler, epoch, total_loss):
    progress_bar = tqdm(range(len(dataloader)))
    progress_bar.set_description(f'loss:{0:>7f}')
    finish_batch_num = (epoch - 1) * len(dataloader)

    # 批量读取数据
    model.train()
    for batch, (X, y) in enumerate(dataloader, start=1):
        X = X["input_ids"].to(device)
        y = y.to(device)
        pred = model(X)
        loss = loss_fn(pred, y)

        # 清楚优化器的梯度
        optimizer.zero_grad()
        # 反向传播损失
        loss.backward()
        # 执行一步优化器
        optimizer.step()
        # 使用学习率调度器更新学习率
        lr_scheduler.step()

        # 更新总损失
        total_loss += loss.item()
        progress_bar.set_description(f'loss:{total_loss / (finish_batch_num + batch):>7f}')
        progress_bar.update(1)
    return total_loss


def test_loop(dataloader, model, mode='Test'):
    assert mode in ['Valid', 'Test']
    size = len(dataloader.dataset)
    correct = 0

    model.eval()
    with torch.no_grad():
        for X, y in dataloader:
            X = X["input_ids"].to(device)
            y = y.to(device)
            pred = model(X)
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        correct /= size
        print(f"{model} Accuracy :{(100*correct):>0.1f}%\n")
        return correct


import torch.optim as optim
from transformers import get_scheduler

learning_rate = 1e-5
epoch_num = 3

loss_fn = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
lr_scheduler = get_scheduler(
    "linear",
    optimizer=optimizer,
    num_warmup_steps=0,
    num_training_steps=epoch_num*len(train_dataloader)
)

total_loss = 0
best_acc = 0.
for t in range(epoch_num):
    print(f'Epoch {t+1}/{epoch_num}\n-------------------------------')
    total_loss = train_loop(train_dataloader, model, loss_fn, optimizer, lr_scheduler, t+1, total_loss)
    valid_acc = test_loop(valid_dataloader, model, mode='Valid')
    if valid_acc > best_acc:
        best_acc = valid_acc
        print('saving new weights...\n')
        torch.save(model.state_dict(), f'epoch_{t+1}_valid_acc_{(100*valid_acc):0.1f}_model_weights.bin')
print("Done!")
