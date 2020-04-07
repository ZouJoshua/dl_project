#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/7/20 6:18 PM
@File    : basic_config.py
@Desc    : 基础配置文件

"""
import configparser
import os

class ConfigBase(object):

    """基础配置参数"""
    def __init__(self, config_file, section=None):
        config_ = configparser.ConfigParser()
        config_.read(config_file)
        if not config_.has_section(section):
            raise Exception("Section={} not found".format(section))

        self.all_params = {}
        for i in config_.items(section):
            self.all_params[i[0]] = i[1]

        self.config = config_[section]
        if not self.config:
            raise Exception("Config file error.")
        # name
        self.model_name = self.config.get("model_name", "model")                                # 模型名称
        # path
        self.data_path = self.config.get("data_path", None)                                     # 数据目录(语料文件\映射文件)
        self.output_path = self.config.get("output_path", None)                                 # 输出目录(模型文件\日志文件\临时文件等)
        self.init_checkpoint_path = self.config.get("init_checkpoint_path", None)               # 预训练bert路径
        self.ckpt_model_path = self.config.get("ckpt_model_path", None)                         # 模型目录
        if not self.ckpt_model_path:
            self.ckpt_model_path = os.path.join(self.output_path, "model")
        os.makedirs(self.ckpt_model_path)
        self.word2idx_file = self.config.get("word2idx_file", None)                             # 词表映射文件
        self.label2idx_file = self.config.get("label2idx_file", None)                           # label映射文件
        self.pretrain_embedding_file = self.config.get("pretrain_embedding_file", None)         # 预训练embedding文件
        self.stopwords_file = self.config.get("stopwords_file", None)                           # 停用词文件
        # model
        self.num_labels = self.config.getint("num_labels", None)                                # 类别数,二分类时置为1,多分类时置为实际类别数
        self.embedding_dim = self.config.getint("embedding_dim", 512)                           # 字\词向量维度
        self.vocab_size = self.config.getint("vocab_size", 20000)                               # 字典(词表)大小
        self.sequence_length = self.config.getint("sequence_length", 512)                       # 序列长度,每句话处理成的长度(短填长切)
        self.learning_rate = self.config.getfloat("learning_rate", 0.001)                       # 学习速率
        self.hidden_size = self.config.getint("hidden_size", 256)                               # 隐藏层单元个数
        self.dropout_keep_prob = self.config.getfloat("dropout_keep_prob", 1.0)                 # 保留神经元的比例,随机失活
        # train,eval,test(predict)
        self.num_epochs = self.config.getint("num_epochs", 1)                                   # 全样本迭代次数
        self.batch_size = self.config.getint("batch_size", 64)                                  # 批样本(mini-batch)大小
        self.eval_every_step = self.config.getint("eval_every_step", 100)                       # 迭代多少step验证一次模型
        self.save_checkpoints_steps = self.config.getint("save_checkpoints_steps", 100)         # 迭代多少step保存一次模型
        self.require_improvement = self.config.getint("require_improvement", 100)               # 若超过100个batch(epoch)验证集指标还没提升，则提前结束训练








file = "/data/work/dl_project/config/dpcnn_pytorch.ini"
config = Config(file, "THUC_NEWS")