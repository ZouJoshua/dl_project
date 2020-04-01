#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/1/20 5:14 PM
@File    : bilstm_attention_model.py
@Desc    : 

"""

import tensorflow as tf

from model_tensorflow.basic_model import BaseModel
import configparser


class Config(object):
    """BiLstm_Attention配置参数"""
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
        self.data_path = config.get("data_path")                           # 数据目录
        self.label2idx_path = config.get("label2idx_path")                 # label映射文件
        self.pretrain_embedding = config.get("pretrain_embedding")         # 预训练词向量文件
        self.stopwords_path = config.get("stopwords_path", "")             # 停用词文件
        self.output_path = config.get("output_path")                       # 输出目录(模型文件\)
        self.ckpt_model_path = config.get("ckpt_model_path", "")           # 模型目录
        self.sequence_length = config.getint("sequence_length")            # 序列长度
        self.num_labels = config.getint("num_labels")                      # 类别数,二分类时置为1,多分类时置为实际类别数
        self.embedding_dim = config.getint("embedding_dim")                # 词向量维度
        self.vocab_size = config.getint("vocab_size")                      # 字典大小
        self.hidden_sizes = eval(config.get("hidden_sizes", "[256,256]"))  # lstm的隐层大小，列表对象，支持多层lstm，只要在列表中添加相应的层对应的隐层大小
        self.output_size = config.getint("output_size")                    # 从高维映射到低维的神经元个数(不设置)
        self.is_training = config.getboolean("is_training", False)
        self.dropout_keep_prob = config.getfloat("dropout_keep_prob")      # 保留神经元的比例
        self.optimization = config.get("optimization", "adam")             # 优化算法
        self.learning_rate = config.getfloat("learning_rate")              # 学习速率
        self.learning_decay_rate = config.getfloat("learning_decay_rate")
        self.learning_decay_steps = config.getint("learning_decay_steps")
        self.l2_reg_lambda = config.getfloat("l2_reg_lambda", 0.0)              # L2正则化的系数，主要对全连接层的参数正则化
        self.max_grad_norm = config.getfloat("max_grad_norm", 5.0)         # 梯度阶段临界值
        self.num_epochs = config.getint("num_epochs")                      # 全样本迭代次数
        self.train_batch_size = config.getint("train_batch_size")          # 训练集批样本大小
        self.eval_batch_size = config.getint("eval_batch_size")            # 验证集批样本大小
        self.test_batch_size = config.getint("test_batch_size")            # 测试集批样本大小
        self.eval_every_step = config.getint("eval_every_step")            # 迭代多少步验证一次模型
        self.model_name = config.get("model_name", "bilstm_attention")              # 模型名称




class BiLstmAttention(BaseModel):

    def __init__(self, config, vocab_size, word_vectors):
        super(BiLstmAttention, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):
        self.embedding_layer()
        self.bi_lstm_layer()
        self.attention_layer()
        self.full_connection_layer()
        self.cal_loss()

        self.predictions = self.get_predictions()
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()

    def embedding_layer(self):
        """
        词嵌入层
        :return:
        """
        with tf.name_scope("embedding-layer"):
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config.embedding_dim],
                                          initializer=tf.contrib.layers.xavier_initializer())

            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，
            # 维度[batch_size, sequence_length, embedding_dim]
            self.embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)
            self._embedded_words = self.embedded_words

    def bi_lstm_layer(self):
        """
        定义双向LSTM的模型结构,利用Bi-LSTM获得上下文的信息，类似于语言模型
        :return:
        """
        with tf.name_scope("bi-lstm-layer"):
            for idx, hidden_size in enumerate(self.config.hidden_sizes):
                with tf.name_scope("bi-lstm" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                    # 定义反向LSTM结构
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                             self.embedded_words, dtype=tf.float32,
                                                                             scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
                    self.embedded_words = tf.concat(outputs, 2)


            # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
            self.fw_output, self.bw_output = tf.split(self.embedded_words, 2, -1)



    def attention_layer(self):
        # 在Bi-LSTM+Attention的论文中，将前向和后向的输出相加
        with tf.name_scope("Attention"):
            lstm_output = self.fw_output + self.bw_output


            # 获得最后一层LSTM的神经元数量
            self.hidden_size = self.config.hidden_sizes[-1]
            # self._output_size = self.config.hidden_sizes[-1]
            # 得到Attention的输出
            self.attention_output = self._attention(lstm_output)


    def _attention(self, H):
        """
        利用Attention机制得到句子的向量表示
        :param H:
        :return:
        """

        # 初始化一个权重向量，是可训练的参数
        attention_w = tf.Variable(tf.random_normal([self.hidden_size], stddev=0.1))

        # 对Bi-LSTM的输出用激活函数做非线性转换
        h_tanh = tf.tanh(H)

        # 对attention_w和h_tanh做矩阵运算，h_tanh=[batch_size, time_step, hidden_size],
        # 计算前做维度转换成[batch_size * time_step, hidden_size]
        # new_h = [batch_size, time_step, 1]，每一个时间步的输出由向量转换成一个数字
        new_h = tf.matmul(tf.reshape(h_tanh, [-1, self.hidden_size]), tf.reshape(attention_w, [-1, 1]))

        # 对new_h做维度转换成[batch_size, time_step]
        restore_h = tf.reshape(new_h, [-1, self.config.sequence_length])

        # 用softmax做归一化处理[batch_size, time_step]
        self.alpha = tf.nn.softmax(restore_h)

        # 利用求得的alpha的值对H进行加权求和，用矩阵运算直接操作
        r = tf.matmul(tf.transpose(H, [0, 2, 1]), tf.reshape(self.alpha, [-1, self.config.sequence_length, 1]))

        # 将三维压缩成二维sequeeze_r=[batch_size, hidden_size]
        sequeeze_r = tf.squeeze(r)

        sentence_repren = tf.tanh(sequeeze_r)

        # 对Attention的输出可以做dropout处理
        output = tf.nn.dropout(sentence_repren, self.keep_prob)

        return output


    def full_connection_layer(self):
        """
        全连接层，后面接dropout以及relu激活
        :return:
        """

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.attention_output, self.keep_prob)


        # 全连接层的输出
        with tf.name_scope("fully_connection_layer"):
            output_w = tf.get_variable(
                "output_w",
                shape=[self.hidden_size, self.config.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_labels]), name="output_b")
            self.logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name="logits")
            # self.logits = tf.matmul(h_drop, output_w) + output_b
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)


    def cal_loss(self):

        with tf.name_scope("loss"):
            # 计算交叉熵损失
            self.labels = tf.cast(self.labels, dtype=tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = tf.reduce_mean(losses)

            # self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.config.l2_reg_lambda
            # self.loss = loss + self.l2_loss
            self.loss = loss + self.config.l2_reg_lambda * self.l2_loss


