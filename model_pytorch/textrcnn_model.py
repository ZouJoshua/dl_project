#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/19/20 3:52 PM
@File    : textrcnn_model.py
@Desc    : 

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pytorch.basic_config import ConfigBase


class Config(ConfigBase):
    """textrcnn_pytorch配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备




class Model(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification
    """
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pretrain_embedding_file is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding_file, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.vocab_size - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout_keep_prob)
        self.maxpool = nn.MaxPool1d(config.sequence_length)
        self.fc = nn.Linear(config.hidden_size * 2 + config.embedding_dim, config.num_labels)

    def forward(self, x):
        x, _ = x
        embed = self.embedding(x)  # [batch_size, seq_len, embeding]=[64, 32, 64]
        out, _ = self.lstm(embed)
        out = torch.cat((embed, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out