#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/13/20 5:48 PM
@File    : bert_pt_model.py
@Desc    : bert 分类模型(pytorch版本)

"""

from torch import nn
import torch
from model_pytorch.bert_common_model import BertConfig, BertModel



class ThucNewsBertModel(nn.Module):
    """使用CLS的方式进行多分类"""
    def __init__(self, config):
        super(ThucNewsBertModel, self).__init__()
        self.bert = BertModel(config)
        self.final_dense = nn.Linear(config.hidden_size, config.num_labels)
        self.activation = nn.Softmax()

    def compute_loss(self, predictions, labels):
        # 将预测和标记的维度展平, 防止出现维度不一致
        # # 逻辑回归
        # predictions = predictions.view(-1)
        # labels = labels.float().view(-1)
        # epsilon = 1e-8
        # # 交叉熵
        # loss =\
        #     - labels * torch.log(predictions + epsilon) - \
        #     (torch.tensor(1.0) - labels) * torch.log(torch.tensor(1.0) - predictions + epsilon)
        # # 求均值, 并返回可以反传的loss
        # # loss为一个实数
        # loss = torch.mean(loss)

        loss = nn.functional.cross_entropy(predictions, labels)

        return loss

    def forward(self, text_input, positional_enc, labels=None):
        encoded_layers, _ = self.bert(text_input, positional_enc,
                                    output_all_encoded_layers=True)

        # encoded_layers是一个含有6个tensor的list,训练只用到第三层,后面参数多未用
        sequence_output = encoded_layers[2]
        # # sequence_output的维度是[batch_size, seq_len, embed_dim]
        # avg_pooled = sequence_output.mean(1)
        # max_pooled = torch.max(sequence_output, dim=1)
        # pooled = torch.cat((avg_pooled, max_pooled[0]), dim=1)

        # 截取#CLS#标签所对应的一条向量, 也就是时间序列维度(seq_len)的第0条
        first_token_tensor = sequence_output[:, 0]

        # 下面是[batch_size, hidden_dim] 到 [batch_size, 1]的映射
        # 这里要解决的是多分类问题
        # predictions = self.dense(first_token_tensor)
        predictions = self.final_dense(first_token_tensor)

        # 用softmax函数做激活, 返回0-1之间的值
        predictions = self.activation(predictions)
        if labels is not None:
            # 计算loss
            loss = self.compute_loss(predictions, labels)
            return predictions, loss
        else:
            return predictions