#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/19/20 3:51 PM
@File    : textrnn_model.py
@Desc    : 

"""

import torch
import torch.nn as nn
import numpy as np

from model_pytorch.basic_config import ConfigBase


class Config(ConfigBase):
    """textrnn_pytorch配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备
        self.hidden_output_size = self.config.getint("hidden_output_size", 64)
        self.num_layers = self.config.getint("num_layers")  # lstm层数


class Model(nn.Module):
    """
    Recurrent Neural Network for Text Classification with Multi-Task Learning
    """
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pretrain_embedding_file is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding_file, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.vocab_size - 1)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout_keep_prob)
        self.fc = nn.Linear(config.hidden_size * 2, config.num_labels)

    def forward(self, x):
        x, _ = x
        out = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        out, _ = self.lstm(out)
        out = self.fc(out[:, -1, :])  # 句子最后时刻的 hidden state
        return out

    '''变长RNN，效果差不多，甚至还低了点...'''
    # def forward(self, x):
    #     x, seq_len = x
    #     out = self.embedding(x)
    #     _, idx_sort = torch.sort(seq_len, dim=0, descending=True)  # 长度从长到短排序（index）
    #     _, idx_unsort = torch.sort(idx_sort)  # 排序后，原序列的 index
    #     out = torch.index_select(out, 0, idx_sort)
    #     seq_len = list(seq_len[idx_sort])
    #     out = nn.utils.rnn.pack_padded_sequence(out, seq_len, batch_first=True)
    #     # [batche_size, seq_len, num_directions * hidden_size]
    #     out, (hn, _) = self.lstm(out)
    #     out = torch.cat((hn[2], hn[3]), -1)
    #     # out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True)
    #     out = out.index_select(0, idx_unsort)
    #     out = self.fc(out)
    #     return out