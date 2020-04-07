#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/19/20 5:20 PM
@File    : bert_rcnn_model.py
@Desc    : 

"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pytorch.pretrained import BertModel, BertTokenizer

import configparser

class Config(object):

    """bert_rcnn_pytorch配置参数"""
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
        self.model_name = config.get("model_name", "bert_rcnn_pytorch")       # 模型名称
        self.data_path = config.get("data_path")                             # 数据目录
        self.output_path = config.get("output_path")                         # 输出目录(模型文件\)
        self.label2idx_path = config.get("label2idx_path")                   # label映射文件
        self.pretrain_bert_path = config.get("pretrain_bert_path", './bert_pretrain')  # 预训练bert路径
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_bert_path)
        self.stopwords_path = config.get("stopwords_path", "")               # 停用词文件
        self.ckpt_model_path = config.get("ckpt_model_path", "")             # 模型目录
        self.sequence_length = config.getint("sequence_length")              # 序列长度,每句话处理成的长度(短填长切)
        self.num_labels = config.getint("num_labels")                        # 类别数,二分类时置为1,多分类时置为实际类别数
        self.embedding_dim = config.getint("embedding_dim")                  # 字\词向量维度
        self.vocab_size = config.getint("vocab_size")                        # 字典大小,词表大小,在运行时赋值
        self.hidden_size = config.getint("hidden_size", 768)                 # lstm隐藏层
        self.rnn_hidden = config.getint("rnn_hidden", 256)
        self.num_layers = config.getint("num_layers", 2)                     # lstm层数
        self.filter_sizes = eval(config.get("filter_size", (2, 3, 4)))       # 卷积核尺寸
        self.num_filters = config.getint("num_filters")                      # 卷积核数量(channels数)
        self.dropout_keep_prob = config.getfloat("dropout_keep_prob", 0.8)   # 保留神经元的比例,随机失活
        self.learning_rate = config.getfloat("learning_rate")                # 学习速率
        self.num_epochs = config.getint("num_epochs")                        # 全样本迭代次数
        self.batch_size = config.getint("batch_size")                        # 批样本大小,mini-batch大小
        self.eval_every_step = config.getint("eval_every_step")              # 迭代多少步验证一次模型
        self.require_improvement = config.getint("require_improvement")      # 若超过1000batch效果还没提升，则提前结束训练
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备



class BertRCNNModel(nn.Module):

    def __init__(self, config):
        super(BertRCNNModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.lstm = nn.LSTM(config.hidden_size, config.rnn_hidden, config.num_layers,
                            bidirectional=True, batch_first=True, dropout=config.dropout_keep_prob)
        self.maxpool = nn.MaxPool1d(config.sequence_length)
        self.fc = nn.Linear(config.rnn_hidden * 2 + config.hidden_size, config.num_labels)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out, _ = self.lstm(encoder_out)
        out = torch.cat((encoder_out, out), 2)
        out = F.relu(out)
        out = out.permute(0, 2, 1)
        out = self.maxpool(out).squeeze()
        out = self.fc(out)
        return out