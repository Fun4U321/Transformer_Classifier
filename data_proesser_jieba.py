#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/9 19:38
# @Author  : Fun.
# @File    : data_proesser_jieba.py
# @Software: PyCharm

import pandas as pd
import jieba

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]
    return stopwords

stopwords = load_stopwords('data/stop_words.txt')

def tokenize_with_jieba(text):
    words = jieba.cut(text)
    result = ' '.join([word for word in words if word not in stopwords])
    return result

def process_csv(input_file, output_file, columns):
    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 对指定列进行分词处理
    for column_name in columns:
        df[column_name] = df[column_name].apply(tokenize_with_jieba)

    # 保存处理后的数据到新的CSV文件
    df.to_csv(output_file, index=False)
    print("处理完成，结果已保存到", output_file)

if __name__ == "__main__":
    # 输入文件名、输出文件名和要处理的列名列表
    input_file = 'data/train.csv'
    output_file = 'data/train_processed_jieba.csv'
    columns = ['Father_node', 'Child_node']

    # 调用处理函数
    process_csv(input_file, output_file, columns)
