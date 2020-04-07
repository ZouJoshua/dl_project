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
import configparser

class Config(object):

    """textrnn_pytorch配置参数"""
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
        self.model_name = config.get("model_name", "textrnn_pytorch")       # 模型名称
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



class TextRNNModel(nn.Module):
    """
    Recurrent Neural Network for Text Classification with Multi-Task Learning
    """
    def __init__(self, config):
        super(TextRNNModel, self).__init__()
        if config.pretrain_embedding is not None:
            self.embedding = nn.Embedding.from_pretrained(config.pretrain_embedding, freeze=False)
        else:
            self.embedding = nn.Embedding(config.vocab_size, config.embed, padding_idx=config.vocab_size - 1)
        self.lstm = nn.LSTM(config.embed, config.hidden_size, config.num_layers,
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