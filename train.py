#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/9 22:19
# @Author  : Fun.
# @File    : train.py
# @Software: PyCharm

import torch
import pandas as pd
import numpy as np
from os import listdir
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import cohen_kappa_score
from my_transformer import build_model
from keras.callbacks import EarlyStopping


# 如果有可用的GPU设备，则在代码中指定使用GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 读取包含分词结果的CSV文件
# df = pd.read_csv('data/train_processed.csv')
df = pd.read_csv('data/train_processed_jieba.csv')
df.dropna(subset=['Father_node'], inplace=True)

# 创建一个Tokenizer对象，限定最大词汇表大小为6000
tok = Tokenizer(num_words=6000, char_level=True)
# 使用fit_on_texts方法来构建词汇表，传入Father_node列的值作为文本列表
tok.fit_on_texts(df['Father_node'].values)
# print("样本数 : ", tok.document_count)
# # 打印词汇表中前10个单词及其索引
# print({k: v for k, v in zip(list(tok.word_index.keys())[:10], list(tok.word_index.values())[:10])})

X = tok.texts_to_sequences(df['Father_node'].values)
# # 查看x的长度的分布
# length = []
# for i in X:
#     length.append(len(i))
# v_c = pd.Series(length).value_counts()
# print(v_c[v_c > 5])  # 频率大于5才展现
# v_c[v_c > 5].plot(kind='bar', figsize=(12, 5))
# plt.show()

# 将序列数据填充成相同长度
X = sequence.pad_sequences(X, maxlen=70)
Y = pd.get_dummies(df['语义关系']).values
dic = {"支持": 1, "反对": 2, "补充": 3, "质疑": 4, "": 0}
dic2 = dict([(value, key) for (key, value) in dic.items()])
Y = df['语义关系'].map(dic).values
# print("X.shape: ", X.shape)
# print("Y.shape: ", Y.shape)

X = np.array(X)
Y = np.array(Y)

# 标签检查
label_counts = df['语义关系'].value_counts()
print(label_counts)

label_4_data = df[df['语义关系'] == 4]
print(label_4_data)

# 检查标签中是否存在空值
# print(df['语义关系'].isnull().sum())

# 重新划分特征和标签
X = sequence.pad_sequences(X, maxlen=70)
Y = df['语义关系'].values

# 划分训练集和测试集
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=0)
X_train.shape, X_test.shape, Y_train.shape, Y_test.shape
Y_test_original = Y_test.copy()
Y_train = to_categorical(Y_train)
Y_test = to_categorical(Y_test)
print(X_train[100:103])
print(Y_test[0])
print(Y_train[0])
Y_test_original[:3]

np.random.seed(0)  # 指定随机数种子
# 单词索引的最大个数6000，单句话最大长度60
top_words = 6000
max_words = 70    # 序列长度
embed_dim = 32    # 嵌入维度
num_labels = 4   # 4分类
batch_size = 32
epochs = 6

def plot_loss(history):
    # 显示训练和验证损失图表
    plt.subplots(1,2,figsize=(10,3))
    plt.subplot(121)
    loss = history.history["loss"]
    epochs = range(1, len(loss)+1)
    val_loss = history.history["val_loss"]
    plt.plot(epochs, loss, "bo", label="Training Loss")
    plt.plot(epochs, val_loss, "r", label="Validation Loss")
    plt.title("Training and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.subplot(122)
    acc = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    plt.plot(epochs, acc, "b-", label="Training Acc")
    plt.plot(epochs, val_acc, "r--", label="Validation Acc")
    plt.title("Training and Validation Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(model, X_test, Y_test_original, sns=None):
    # 预测概率
    prob = model.predict(X_test)
    # 预测类别
    pred = np.argmax(prob, axis=1)
    # 数据透视表，混淆矩阵
    pred = pd.Series(pred).map(dic2)
    Y_test_original = pd.Series(Y_test_original).map(dic2)
    table = pd.crosstab(Y_test_original, pred, rownames=['Actual'], colnames=['Predicted'])
    # print(table)
    sns.heatmap(table, cmap='Blues', fmt='.20g', annot=True)
    plt.tight_layout()
    plt.show()
    # 计算混淆矩阵的各项指标
    print(classification_report(Y_test_original, pred))
    # 科恩Kappa指标
    print('科恩Kappa'+str(cohen_kappa_score(Y_test_original, pred)))

# 定义训练函数
def train_fuc(max_words = max_words, mode = 'BiLSTM+Attention', batch_size = 32, epochs = 10, hidden_dim = [32],
              show_loss=True, show_confusion_matrix=True):
    # 构建模型
    model = build_model(max_words=max_words, mode=mode)
    print(model.summary())
    es = EarlyStopping(patience=5)
    history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2, verbose=1,
                        callbacks=[es])
    print('——————————-----------------——训练完毕—————-----------------------------———————')
    # 评估模型
    loss, accuracy = model.evaluate(X_test, Y_test)
    print("测试数据集的准确度 = {:.4f}".format(accuracy))

    if show_loss:
        plot_loss(history)
    if show_confusion_matrix:
        plot_confusion_matrix(model=model, X_test=X_test, Y_test_original=Y_test_original)

print("Y_train shape:", Y_train.shape)
print("Y_test shape:", Y_test.shape)

train_fuc(mode='MLP', batch_size=batch_size, epochs=epochs)