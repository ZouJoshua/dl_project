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
import numpy as np
import configparser


class Config(object):

    """textcnn_pytorch配置参数"""
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
        self.model_name = config.get("model_name", "textcnn_pytorch")       # 模型名称
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
        self.filter_sizes = eval(config.get("filter_size", (2, 3, 4)))       # 卷积核尺寸
        self.num_filters = config.getint("num_filters")                      # 卷积核数量(channels数)
        self.dropout_keep_prob = config.getfloat("dropout_keep_prob", 0.8)   # 保留神经元的比例,随机失活
        self.learning_rate = config.getfloat("learning_rate")                # 学习速率
        self.num_epochs = config.getint("num_epochs")                        # 全样本迭代次数
        self.batch_size = config.getint("batch_size")                        # 批样本大小,mini-batch大小
        self.eval_every_step = config.getint("eval_every_step")              # 迭代多少步验证一次模型
        self.require_improvement = config.getint("require_improvement")      # 若超过1000batch效果还没提升，则提前结束训练
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备






class TextCNNModel(nn.Module):
    """
    Convolutional Neural Networks for Sentence Classification
    """
    def __init__(self, config):
        super(TextCNNModel, self).__init__()
        if config.pretrain_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embedding_dim, padding_idx=config.vocab_size - 1)
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embedding_dim)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout_keep_prob)
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x[0])
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        return out