#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/9 13:22
# @Author  : Fun.
# @File    : data_processer.py
# @Software: PyCharm

import pandas as pd
import pkuseg

def load_stopwords(stopwords_file):
    with open(stopwords_file, 'r', encoding='utf-8') as f:
        stopwords = [line.strip() for line in f]
    return set(stopwords)

def tokenize_with_pkuseg(text, stopwords):
    seg = pkuseg.pkuseg(user_dict='data/user_dic.txt', model_name='medicine')
    words = seg.cut(text)
    result = ' '.join([word for word in words if word not in stopwords])
    return result

def process_csv(input_file, output_file, columns, stopwords_file):
    # 加载停用词典
    stopwords = load_stopwords(stopwords_file)

    # 读取CSV文件
    df = pd.read_csv(input_file)

    # 对指定列进行分词处理并过滤停用词
    for column_name in columns:
        df[column_name] = df[column_name].apply(lambda x: tokenize_with_pkuseg(x, stopwords))

    # 保存处理后的数据到新的CSV文件
    df.to_csv(output_file, index=False)
    print("处理完成，结果已保存到", output_file)


if __name__ == "__main__":
    # 输入文件名、输出文件名、停用词文件名和要处理的列名列表
    input_file = 'data/train.csv'
    output_file = 'data/train_processed.csv'
    columns = ['Father_node', 'Child_node']
    stopwords_file = 'data/stop_words.txt'

    # 调用处理函数
    process_csv(input_file, output_file, columns, stopwords_file)
