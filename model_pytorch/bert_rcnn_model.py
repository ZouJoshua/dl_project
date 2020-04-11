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
from model_pytorch.basic_config import ConfigBase

class Config(ConfigBase):
    """bert_rcnn_pytorch配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 设备
        self.tokenizer = BertTokenizer.from_pretrained(self.init_checkpoint_path)
        self.rnn_hidden = self.config.getint("rnn_hidden", 256)
        self.num_layers = self.config.getint("num_layers", 2)  # lstm层数
        self.filter_sizes = eval(self.config.get("filter_size", (2, 3, 4)))  # 卷积核尺寸
        self.num_filters = self.config.getint("num_filters")  # 卷积核数量(channels数)



class BertRCNNModel(nn.Module):

    def __init__(self, config):
        super(BertRCNNModel, self).__init__()
        self.bert = BertModel.from_pretrained(config.init_checkpoint_path)
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