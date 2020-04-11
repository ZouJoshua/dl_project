#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/19/20 5:02 PM
@File    : bert_cnn_model.py
@Desc    : 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from model_pytorch.pretrained import BertModel, BertTokenizer
from model_pytorch.basic_config import ConfigBase


class Config(ConfigBase):
    """bert_cnn_pytorch配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备
        self.tokenizer = BertTokenizer.from_pretrained(self.init_checkpoint_path)
        self.filter_sizes = eval(self.config.get("filter_size", (2, 3, 4)))       # 卷积核尺寸
        self.num_filters = self.config.getint("num_filters")                      # 卷积核数量(channels数)


class BertCNNModel(nn.Module):

    def __init__(self, config):
        super(BertCNNModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.init_checkpoint_path)
        for param in self.bert.parameters():
            param.requires_grad = True
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.hidden_size)) for k in config.filter_sizes])
        self.dropout = nn.Dropout(config.dropout_keep_prob)

        self.fc_cnn = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_labels)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        context = x[0]  # 输入的句子
        mask = x[2]  # 对padding部分进行mask，和句子一个size，padding部分用0表示，如：[1, 1, 1, 1, 0, 0]
        encoder_out, text_cls = self.bert(context, attention_mask=mask, output_all_encoded_layers=False)
        out = encoder_out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc_cnn(out)
        return out