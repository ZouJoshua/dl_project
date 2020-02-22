#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2/20/20 3:29 PM
@File    : transformer_model.py
@Desc    : transformer 模型学习

"""



import numpy as np
import matplotlib as plt
import seaborn as sns
import math


def get_positional_encoding(max_seq_len, embed_dim):
    """
    初始化一个positional encoding
    :param max_seq_len: 最大的序列长度
    :param embed_dim: 字嵌入的维度
    :return:
    """
    positional_encoding = np.array([[pos / np.power(10000, 2 * i / embed_dim) for i in range(embed_dim)]
                                    if pos != 0 else np.zeros(embed_dim) for pos in range(max_seq_len)])
    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])
    positional_encoding[1:, 1::2] = np.sin(positional_encoding[1:, 1::2])
    # 归一化,用位置嵌入的每一行除以它的模长
    # denominator = np.sqrt(np.sum(position_enc**2, axis=1, keepdims=True))
    # position_enc = position_enc / (denominator +1e-8)
    return positional_encoding

