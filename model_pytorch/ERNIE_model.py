#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/19/20 4:53 PM
@File    : ERNIE_model.py
@Desc    : 

"""

import torch
import torch.nn as nn
from model_pytorch.pretrained import BertModel, BertTokenizer
import configparser

class Config(object):

    """ERNIE配置参数"""
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
        self.model_name = config.get("model_name", "ERNIE_pytorch")       # 模型名称
        self.data_path = config.get("data_path")                             # 数据目录
        self.output_path = config.get("output_path")                         # 输出目录(模型文件\)
        self.label2idx_path = config.get("label2idx_path")                   # label映射文件
        self.pretrain_bert_path = config.get("pretrain_bert_path", './ERNIE_pretrain')  # 预训练bert路径
        self.tokenizer = BertTokenizer.from_pretrained(self.pretrain_bert_path)
        self.stopwords_path = config.get("stopwords_path", "")               # 停用词文件
        self.ckpt_model_path = config.get("ckpt_model_path", "")             # 模型目录
        self.sequence_length = config.getint("sequence_length")              # 序列长度,每句话处理成的长度(短填长切)
        self.num_labels = config.getint("num_labels")                        # 类别数,二分类时置为1,多分类时置为实际类别数
        self.hidden_size = config.getint("hidden_size", 768)                 # 隐藏层大小
        self.dropout_keep_prob = config.getfloat("dropout_keep_prob", 0.8)   # 保留神经元的比例,随机失活
        self.learning_rate = config.getfloat("learning_rate", 5e-5)                # 学习速率
        self.num_epochs = config.getint("num_epochs")                        # 全样本迭代次数
        self.batch_size = config.getint("batch_size")                        # 批样本大小,mini-batch大小
        self.eval_every_step = config.getint("eval_every_step")              # 迭代多少步验证一次模型
        self.require_improvement = config.getint("require_improvement")      # 若超过1000batch效果还没提升，则提前结束训练
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # 设备


class ERNIEModel(nn.Module):

    def __init__(self, config):
        super(ERNIEModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.pretrain_bert_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.fc = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        _, pooled = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = self.fc(pooled)
        return out