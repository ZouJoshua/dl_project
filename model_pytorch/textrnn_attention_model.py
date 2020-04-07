#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/19/20 3:54 PM
@File    : textrnn_attention_model.py
@Desc    : 

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pytorch.basic_config import ConfigBase


class Config(ConfigBase):
    """textrnn_attention_pytorch配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)
        self.hidden_output_size = self.config.getint("hidden_output_size", 64)
        self.num_layers = self.config.getint("num_layers")  # lstm层数




class Model(nn.Module):
    """
    Attention-Based Bidirectional Long Short-Term Memory Networks for Relation Classification
    """
    def __init__(self, config):
        super(Model, self).__init__()
        if config.pretrain_embedding_file is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding_file, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.vocab_size - 1)
        self.lstm = nn.LSTM(config.embedding_dim, config.hidden_size, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout_keep_prob)
        self.tanh1 = nn.Tanh()
        # self.u = nn.Parameter(torch.Tensor(config.hidden_size * 2, config.hidden_size * 2))
        self.w = nn.Parameter(torch.Tensor(config.hidden_size * 2))
        self.tanh2 = nn.Tanh()
        self.fc1 = nn.Linear(config.hidden_size * 2, config.hidden_output_size)
        self.fc = nn.Linear(config.hidden_output_size, config.num_labels)

    def forward(self, x):
        x, _ = x
        emb = self.embedding(x)  # [batch_size, seq_len, embeding]=[128, 32, 300]
        H, _ = self.lstm(emb)  # [batch_size, seq_len, hidden_size * num_direction]=[128, 32, 256]

        M = self.tanh1(H)  # [128, 32, 256]
        # M = torch.tanh(torch.matmul(H, self.u))
        alpha = F.softmax(torch.matmul(M, self.w), dim=1).unsqueeze(-1)  # [128, 32, 1]
        out = H * alpha  # [128, 32, 256]
        out = torch.sum(out, 1)  # [128, 256]
        out = F.relu(out)
        out = self.fc1(out)
        out = self.fc(out)  # [128, 64]
        return out
