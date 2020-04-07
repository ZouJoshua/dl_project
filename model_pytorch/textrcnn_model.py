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
import numpy as np
import configparser


class Config(object):

    """textrcnn_pytorch配置参数"""
    def __init__(self, config_file, section=None):
        config_ = configparser.ConfigParser()
        config_.read(config_file)
        if not config_.has_section(section):
            raise Exception("Section={} not found".format(section))

        self.all_params = {}
        for i in config_.items(section):
            self.all_params[i[0]] = i[1]

        config = config_[section]
        if not config:
            raise Exception("Config file error.")
        self.model_name = config.get("model_name", "textrcnn_pytorch")       # 模型名称
        self.data_path = config.get("data_path")                             # 数据目录
        self.output_path = config.get("output_path")                         # 输出目录(模型文件\)
        self.label2idx_path = config.get("label2idx_path")                   # label映射文件
        self.pretrain_embedding = config.get("pretrain_embedding")           # 预训练词向量文件
        self.stopwords_path = config.get("stopwords_path", "")               # 停用词文件
        self.ckpt_model_path = config.get("ckpt_model_path", "")             # 模型目录
        self.sequence_length = config.getint("sequence_length")              # 序列长度,每句话处理成的长度(短填长切)
        self.num_labels = config.getint("num_labels")                        # 类别数,二分类时置为1,多分类时置为实际类别数
        self.embedding_dim = config.getint("embedding_dim")                  # 字\词向量维度
        self.vocab_size = config.getint("vocab_size")                        # 字典大小,词表大小,在运行时赋值
        self.hidden_size = config.getint("hidden_size", 128)                 # lstm隐藏层
        self.num_layers = config.getint("num_layers")                        # lstm层数
        self.dropout_keep_prob = config.getfloat("dropout_keep_prob", 0.8)   # 保留神经元的比例,随机失活
        self.learning_rate = config.getfloat("learning_rate")                # 学习速率
        self.num_epochs = config.getint("num_epochs")                        # 全样本迭代次数
        self.batch_size = config.getint("batch_size")                        # 批样本大小,mini-batch大小
        self.eval_every_step = config.getint("eval_every_step")              # 迭代多少步验证一次模型
        self.require_improvement = config.getint("require_improvement")      # 若超过1000batch效果还没提升，则提前结束训练
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备




class TextRCNNModel(nn.Module):
    """
    Recurrent Convolutional Neural Networks for Text Classification
    """
    def __init__(self, config):
        super(TextRCNNModel, self).__init__()
        if config.pretrain_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding, freeze=False)
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