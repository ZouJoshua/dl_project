#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/19/20 3:45 PM
@File    : fasttext_model.py
@Desc    : 

"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pytorch.basic_config import ConfigBase


class Config(ConfigBase):
    """fasttext_pytorch配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)
        self.ngram_vocab_size = self.config.getint("ngram_vocab_size", 200000)  # ngram 词表大小


class FasttextModel(nn.Module):
    """
    Bag of Tricks for Efficient Text Classification
    """
    def __init__(self, config):
        super(FasttextModel, self).__init__()
        if config.pretrain_embedding_file is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding_file, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.vocab_size - 1)
        self.embedding_ngram2 = nn.Embedding(config.ngram_vocab_size, config.embedding_dim)
        self.embedding_ngram3 = nn.Embedding(config.ngram_vocab_size, config.embedding_dim)
        self.dropout = nn.Dropout(config.dropout_keep_prob)
        self.fc1 = nn.Linear(config.embedding_dim * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):

        out_word = self.embedding(x[0])
        out_bigram = self.embedding_ngram2(x[2])
        out_trigram = self.embedding_ngram3(x[3])
        out = torch.cat((out_word, out_bigram, out_trigram), -1)

        out = out.mean(dim=1)
        out = self.dropout(out)
        out = self.fc1(out)
        out = F.relu(out)
        out = self.fc2(out)
        return out