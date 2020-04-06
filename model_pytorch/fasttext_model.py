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
import numpy as np
import configparser


class Config(object):

    """fasttext配置参数"""
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
        self.model_name = config.get("model_name", "fasttext_pytorch")
        self.data_path = config.get("data_path")  # 数据目录
        self.label2idx_path = config.get("label2idx_path")  # label映射文件
        self.pretrain_embedding = config.get("pretrain_embedding")  # 预训练词向量文件
        self.stopwords_path = config.get("stopwords_path", "")  # 停用词文件
        self.output_path = config.get("output_path")  # 输出目录(模型文件\)
        self.ckpt_model_path = config.get("ckpt_model_path", "")  # 模型目录
        self.sequence_length = config.getint("sequence_length")  # 序列长度
        self.num_labels = config.getint("num_labels")  # 类别数,二分类时置为1,多分类时置为实际类别数
        self.embedding_dim = config.getint("embedding_dim")  # 词向量维度
        self.vocab_size = config.getint("vocab_size")  # 字典大小
        self.output_size = config.getint("output_size")  # 从高维映射到低维的神经元个数(不设置)
        self.is_training = config.getboolean("is_training", False)
        self.dropout_keep_prob = config.getfloat("dropout_keep_prob", 0.8)  # 保留神经元的比例
        self.learning_rate = config.getfloat("learning_rate")  # 学习速率
        self.num_epochs = config.getint("num_epochs")  # 全样本迭代次数
        self.train_batch_size = config.getint("train_batch_size")  # 训练集批样本大小
        self.eval_batch_size = config.getint("eval_batch_size")  # 验证集批样本大小
        self.test_batch_size = config.getint("test_batch_size")  # 测试集批样本大小
        self.eval_every_step = config.getint("eval_every_step")  # 迭代多少步验证一次模型
        self.require_improvement = config.getint("require_improvement")  # 若超过1000batch效果还没提升，则提前结束训练

    def __init__(self, dataset, embedding):
        self.model_name = 'FastText'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 词表
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型训练结果
        self.log_path = dataset + '/log/' + self.model_name
        self.embedding_pretrained = torch.tensor(
            np.load(dataset + '/data/' + embedding)["embeddings"].astype('float32'))\
            if embedding != 'random' else None                                       # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备

        self.dropout = 0.5                                              # 随机失活
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.n_vocab = 0                                                # 词表大小，在运行时赋值
        self.num_epochs = 20                                            # epoch数
        self.batch_size = 128                                           # mini-batch大小
        self.pad_size = 32                                              # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                       # 学习率
        self.embed = self.embedding_pretrained.size(1)\
            if self.embedding_pretrained is not None else 300           # 字向量维度
        self.hidden_size = 256                                          # 隐藏层大小
        self.n_gram_vocab = 250499                                      # ngram 词表大小


'''Bag of Tricks for Efficient Text Classification'''


class FasttextModel(nn.Module):
    def __init__(self, config):
        super(FasttextModel, self).__init__()
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        self.embedding_ngram2 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.embedding_ngram3 = nn.Embedding(config.n_gram_vocab, config.embed)
        self.dropout = nn.Dropout(config.dropout)
        self.fc1 = nn.Linear(config.embed * 3, config.hidden_size)
        # self.dropout2 = nn.Dropout(config.dropout)
        self.fc2 = nn.Linear(config.hidden_size, config.num_classes)

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