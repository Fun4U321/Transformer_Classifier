#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/10 11:45
# @Author  : Fun.
# @File    : my_transformer.py
# @Software: PyCharm

from tensorflow.keras import layers
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.preprocessing import sequence
from keras.models import Sequential,Model
from keras.layers import Dense,Input, Dropout, Embedding, Flatten,MaxPooling1D,Conv1D,SimpleRNN,LSTM,GRU,Multiply,GlobalMaxPooling1D
from keras.layers import Bidirectional,Activation,BatchNormalization,GlobalAveragePooling1D,MultiHeadAttention
from keras.layers.merge import concatenate

np.random.seed(0)  # 指定随机数种子
# 单词索引的最大个数6000，单句话最大长度60
top_words = 6000
max_words = 70    # 序列长度
embed_dim = 32    # 嵌入维度
num_labels = 4   # 4分类

class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim
        self.dense_dim = dense_dim
        self.num_heads = num_heads
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"), layers.Dense(embed_dim), ])
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
        attention_output = self.attention(inputs, inputs, attention_mask=mask)
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim, })
        return config

    # 定义一个位置编码的嵌入层。
class PositionalEmbedding(layers.Layer):
    def __init__(self, sequence_length, input_dim, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.token_embeddings = layers.Embedding(input_dim=input_dim, output_dim=output_dim)
        self.position_embeddings = layers.Embedding(input_dim=sequence_length, output_dim=output_dim)
        self.sequence_length = sequence_length
        self.input_dim = input_dim
        self.output_dim = output_dim

    def call(self, inputs):
        length = tf.shape(inputs)[-1]
        positions = tf.range(start=0, limit=length, delta=1)
        embedded_tokens = self.token_embeddings(inputs)
        embedded_positions = self.position_embeddings(positions)
        return embedded_tokens + embedded_positions

    def compute_mask(self, inputs, mask=None):
        return tf.math.not_equal(inputs, 0)

    def get_config(self):
        config = super().get_config()
        config.update({
            "output_dim": self.output_dim,
            "sequence_length": self.sequence_length,
            "input_dim": self.input_dim, })
        return config

def build_model(top_words=top_words, max_words=max_words, num_labels=num_labels, mode='LSTM', hidden_dim=[64]):
    if mode == 'RNN':
        model = Sequential()
        model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True))
        model.add(Dropout(0.25))
        model.add(SimpleRNN(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))
    elif mode == 'MLP':
        model = Sequential()
        model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(hidden_dim[0], activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))
    elif mode == 'LSTM':
        model = Sequential()
        model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim))
        model.add(Dropout(0.25))
        model.add(LSTM(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))
    elif mode == 'GRU':
        model = Sequential()
        model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim))
        model.add(Dropout(0.25))
        model.add(GRU(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))
    elif mode == 'CNN':  # 一维卷积
        model = Sequential()
        model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(Flatten())
        model.add(Dense(hidden_dim[0], activation="relu"))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))
    elif mode == 'CNN+LSTM':
        model = Sequential()
        model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim))
        model.add(Dropout(0.25))
        model.add(Conv1D(filters=32, kernel_size=3, padding="same", activation="relu"))
        model.add(MaxPooling1D(pool_size=2))
        model.add(LSTM(hidden_dim[0]))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation="softmax"))
    elif mode == 'BiLSTM':
        model = Sequential()
        model.add(Embedding(top_words, input_length=max_words, output_dim=embed_dim))
        model.add(Bidirectional(LSTM(64)))
        model.add(Dense(hidden_dim[0], activation='relu'))
        model.add(Dropout(0.25))
        model.add(Dense(num_labels, activation='softmax'))
    # 下面的网络采用Funcional API实现
    elif mode == 'TextCNN':
        inputs = Input(name='inputs', shape=[max_words, ], dtype='float64')
        ## 词嵌入使用预训练的词向量
        layer = Embedding(top_words, input_length=max_words, output_dim=embed_dim)(inputs)
        ## 词窗大小分别为3,4,5
        cnn1 = Conv1D(32, 3, padding='same', strides=1, activation='relu')(layer)
        cnn1 = MaxPooling1D(pool_size=2)(cnn1)
        cnn2 = Conv1D(32, 4, padding='same', strides=1, activation='relu')(layer)
        cnn2 = MaxPooling1D(pool_size=2)(cnn2)
        cnn3 = Conv1D(32, 5, padding='same', strides=1, activation='relu')(layer)
        cnn3 = MaxPooling1D(pool_size=2)(cnn3)
        # 合并三个模型的输出向量
        cnn = concatenate([cnn1, cnn2, cnn3], axis=-1)
        x = Flatten()(cnn)
        x = Dense(hidden_dim[0], activation='relu')(x)
        output = Dense(num_labels, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=output)

    elif mode == 'Attention':
        inputs = Input(name='inputs', shape=[max_words, ], dtype='float64')
        x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
        x = MultiHeadAttention(1, key_dim=embed_dim)(x, x, x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(num_labels, activation='softmax')(x)
        model = Model(inputs=[inputs], outputs=output)

    elif mode == 'MultiHeadAttention':
        inputs = Input(name='inputs', shape=[max_words, ], dtype='float64')
        x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
        x = MultiHeadAttention(8, key_dim=embed_dim)(x, x, x)
        x = GlobalAveragePooling1D()(x)
        x = Dropout(0.2)(x)
        x = Dense(32, activation='relu')(x)
        output = Dense(num_labels, activation='softmax')(x)
        model = Model(inputs=[inputs], outputs=output)

    elif mode == 'Attention+BiLSTM':
        inputs = Input(name='inputs', shape=[max_words, ], dtype='float64')
        x = Embedding(top_words, input_length=max_words, output_dim=embed_dim)(inputs)
        x = MultiHeadAttention(2, key_dim=embed_dim)(x, x, x)
        x = Bidirectional(LSTM(hidden_dim[0]))(x)
        x = Dense(64, activation='relu')(x)
        x = Dropout(0.2)(x)
        output = Dense(num_labels, activation='softmax')(x)
        model = Model(inputs=inputs, outputs=output)

    elif mode == 'BiGRU+Attention':
        inputs = Input(name='inputs', shape=[max_words, ], dtype='float64')
        x = Embedding(top_words, input_length=max_words, output_dim=embed_dim)(inputs)
        x = Bidirectional(GRU(32, return_sequences=True))(x)
        x = MultiHeadAttention(2, key_dim=embed_dim)(x, x, x)
        x = Bidirectional(GRU(32))(x)
        x = Dropout(0.2)(x)
        output = Dense(num_labels, activation='softmax')(x)
        model = Model(inputs=[inputs], outputs=output)

    elif mode == 'Transformer':
        inputs = Input(name='inputs', shape=[max_words, ], dtype='float64')
        x = Embedding(top_words, input_length=max_words, output_dim=embed_dim, mask_zero=True)(inputs)
        x = TransformerEncoder(embed_dim, 32, 4)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_labels, activation='softmax')(x)
        model = Model(inputs, outputs)

    elif mode == 'PositionalEmbedding+Transformer':
        inputs = Input(name='inputs', shape=[max_words, ], dtype='float64')
        x = PositionalEmbedding(sequence_length=max_words, input_dim=top_words, output_dim=embed_dim)(inputs)
        x = TransformerEncoder(embed_dim, 32, 4)(x)
        x = GlobalMaxPooling1D()(x)
        x = Dropout(0.5)(x)
        outputs = Dense(num_labels, activation='softmax')(x)
        model = Model(inputs, outputs)

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    return model