#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/27 13:08
# @Author  : Fun.
# @File    : Siamese Transformer.py
# @Software: PyCharm

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# 多头注意力机制。Q，K，V通过线性变换后分别传入多个注意力头，然后计算注意力权重，并进行加权求和。
# 最后输出经过线性变换得到的注意力输出。
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads):
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.d_v = d_model // n_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V, mask=None):
        Q = self.W_Q(Q)
        K = self.W_K(K)
        V = self.W_V(V)
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        scores = torch.matmul(Q, K.transpose(-1, -2)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, V)
        attn_output = self.combine_heads(attn_output)
        attn_output = self.W_O(attn_output)
        return attn_output

    def split_heads(self, x):
        batch_size, seq_len, d_model = x.size()
        x = x.view(batch_size, seq_len, self.n_heads, self.d_k)
        return x.transpose(1, 2)

    def combine_heads(self, x):
        batch_size, n_heads, seq_len, d_v = x.size()
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch_size, seq_len, n_heads * d_v)
        return x


# 位置编码。通过计算位置编码向量，为输入序列中的每个位置添加不同的位置信息。
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=0.1)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x * math.sqrt(self.d_model)
        x = x + self.pe[:, :x.size(1)]
        x = self.dropout(x)
        return x


# 前馈神经网络。包含两个线性层和ReLU激活函数，用于对输入进行非线性变化。
class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        return x


# 编码器层，包含了一个多头注意力机制和一个前馈神经网络模块，并通过LayerNorm和Dropout进行正则化和残差连接。
class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff):
        super(EncoderLayer, self).__init__()
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(p=0.1)
        self.dropout2 = nn.Dropout(p=0.1)

    def forward(self, x, mask=None):
        attn_output = self.multi_head_attn(x, x, x, mask=mask)
        x = x + self.dropout1(attn_output)
        x = self.layer_norm1(x)
        ff_output = self.feed_forward(x)
        x = x + self.dropout2(ff_output)
        x = self.layer_norm2(x)
        return x


# 编码器。由多个编码器层组成，包含了词嵌入层和位置编码层。
class Encoder(nn.Module):
    def __init__(self, input_size, d_model, n_heads, d_ff, n_layers):
        super(Encoder, self).__init__()
        self.embedding = nn.Embedding(input_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderLayer(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        for layer in self.layers:
            x = layer(x, mask=mask)
        x = self.layer_norm(x)
        return x


# 模型定义。预训练模型仅仅被用作编码器，模型中还会包含很多自定义的模块。
class Transformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, n_heads, d_ff, n_layers):
        super(Transformer, self).__init__()
        self.encoder = Encoder(input_size, d_model, n_heads, d_ff, n_layers)
        self.output_layer = nn.Linear(d_model, output_size)

    def forward(self, x, mask=None):
        x = self.encoder(x, mask)
        x = x[:, 0, :]
        x = self.output_layer(x)
        return x


class SiameseTransformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, n_heads, d_ff, n_layers):
        super(SiameseTransformer, self).__init__()
        self.transformer1 = Transformer(input_size, d_model, n_heads, d_ff, n_layers)
        self.transformer2 = Transformer(input_size, d_model, n_heads, d_ff, n_layers)
        self.fc = nn.Linear(d_model * 2, output_size)  # 全连接层将两个句子的表示组合在一起

    def forward(self, x1, x2, mask=None):
        # 分别经过两个Transformer编码器
        x1 = self.transformer1(x1, mask)
        x2 = self.transformer2(x2, mask)

        # 将两个句子的表示连接在一起
        x = torch.cat((x1, x2), dim=1)

        # 经过全连接层得到最终的输出
        x = self.fc(x)
        return x
