#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 15:34
# @Author  : Fun.
# @File    : dataset.py
# @Software: PyCharm

import pandas as pd
from torch.utils.data import Dataset

class MyDataset(Dataset):
    # 定义类的构造函数，接受csv文件路径
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)
        self.label_map = {"体育": 0, "家居": 1, "房产": 2, "教育": 3, "财经": 4}

    # 返回数据集的长度，即样本的数量。
    def __len__(self):
        return len(self.data)

    # 接受一个索引‘idx’作为参数，用于获取数据集中特定索引的样本
    def __getitem__(self, idx):
        # 根据给定的索引‘idx’，从csv文件中获取对应的数据，并将文本数据和标签分别存储在字典项‘text’和‘label’中
        sample = {'text': self.data.iloc[idx, 1], 'label': self.label_map[self.data.iloc[idx, 0]]}
        return sample
