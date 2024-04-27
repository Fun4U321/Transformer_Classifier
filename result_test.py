#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/27 10:52
# @Author  : Fun.
# @File    : result_test.py
# @Software: PyCharm
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from models import Transformer
from dataloder import MyDataLoader

# 检查GPU是否可用
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 加载模型
input_size = 21128
output_size = 5
d_model = 128
n_heads = 8
d_ff = 2048
n_layers = 6
model = Transformer(input_size=input_size, output_size=output_size, d_model=d_model, n_heads=n_heads,
                    d_ff=d_ff, n_layers=n_layers).to(device)
model.load_state_dict(torch.load('epoch_1_valid_acc_20.0_model_weights.bin'))  # 替换为你的模型权重文件路径
model.eval()

# 加载数据集和数据加载器
test_dataloader = MyDataLoader(csv_file='dataset/THUNews/5_5000/dev.csv', batch_size=4, max_length=256)\
    .get_data_loader()

# 统计所有预测结果中最常见的类别
predicted_classes = []

with torch.no_grad():
    for X, y in test_dataloader:  # y 中存储了标签
        X = X["input_ids"].to(device)
        pred = model(X)
        # 获取预测结果中概率最高的类别
        batch_predicted_classes = torch.argmax(pred, dim=1).tolist()
        predicted_classes.extend(batch_predicted_classes)
        # 打印每次预测的结果
        print("Batch Predictions:", batch_predicted_classes)

# 找到所有预测结果中最常见的类别
most_common_class = max(set(predicted_classes), key=predicted_classes.count)

# 检查所有预测结果是否都属于同一类别
all_same_class = all(pred_class == most_common_class for pred_class in predicted_classes)

# 打印结果
if all_same_class:
    print("所有预测结果都属于同一类别：", most_common_class)
else:
    print("预测结果不全属于同一类别。")
