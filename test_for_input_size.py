#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/4/20 18:07
# @Author  : Fun.
# @File    : test.py
# @Software: PyCharm

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("bert-base-chinese")
vocab_size = len(tokenizer)
print("Vocabulary size:", vocab_size)
