#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 16:24
# @Author  : Fun.
# @File    : dataloder.py
# @Software: PyCharm

# 创建好数据集之后，需要通过datalodaer来按批加载数据集，将样本组织成模型可以接受的输入格式。
# 对于NLP任务来说，这个环节就是对一个batch中的句子按照预训练模型的要求进行编码（包括padding、截断等操作）

import torch
from torch.utils.data import DataLoader
# 自动选择和加载适合预训练模型的分词器
from transformers import AutoTokenizer
from dataset import MyDataset


class MyDataLoader:
    # 定义类的构造函数
    def __init__(self, csv_file, batch_size, max_length):
        self.dataset = MyDataset(csv_file)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
        self.batch_size = batch_size
        self.max_length = max_length

    # 用于对一个batch中的样本进行预处理和组织
    def collate_fn(self, batch):
        # 从batch中提取所有的样本文本数据，存储到texts列表中
        texts = [item['text'] for item in batch]
        labels = [item['label'] for item in batch]
        # 使用AutoTokenizer对象对所有文本数据进行分词和预处理
        X = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt")
        y = torch.tensor(labels)
        return X, y
    # 定义get_data_loader方法，用于创建并返回数据加载器对象

    def get_data_loader(self):
        return DataLoader(self.dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)