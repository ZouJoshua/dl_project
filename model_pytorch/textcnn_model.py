#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/19/20 3:48 PM
@File    : textcnn_model.py
@Desc    : 

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pytorch.basic_config import ConfigBase


class Config(ConfigBase):
    """textcnn_pytorch配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备
        # self.device = torch.device("cpu")  # 设备
        self.filter_sizes = eval(self.config.get("filter_sizes", (2, 3, 4)))       # 卷积核尺寸
        self.num_filters = self.config.getint("num_filters")                      # 卷积核数量(channels数)


class Model(nn.Module):
    """
    Convolutional Neural Networks for Sentence Classification
    """
    def __init__(self, config, pretrain_embedding=None):
        super(Model, self).__init__()
        if pretrain_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(torch.tensor(pretrain_embedding), freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.vocab_size - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout_keep_prob)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)  # len(filter_sizes)*(N,num_filters,W)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)  # len(filter_sizes)*(N,num_filters)
        return x

    def forward(self, x):
        out = self.embedding(x[0])  # (N,W,D)
        out = out.unsqueeze(1)  # (N,channel,W,D)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)  #(N,num_filters*len(filter_sizes))
        out = self.dropout(out)
        logit = self.fc(out)
        return logit