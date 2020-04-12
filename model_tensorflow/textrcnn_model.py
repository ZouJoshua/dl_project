#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-3 下午5:05
@File    : textrcnn_model.py
@Desc    : 
"""

import tensorflow as tf

from model_tensorflow.basic_model import BaseModel
from model_tensorflow.basic_config import ConfigBase


class Config(ConfigBase):
    """RCNN配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)
        self.hidden_sizes = eval(self.config.get("hidden_sizes", "[256,256]"))  # lstm的隐层大小，列表对象，支持多层lstm，只要在列表中添加相应的层对应的隐层大小



class RCNN(BaseModel):

    def __init__(self, config, vocab_size, word_vectors):
        super(RCNN, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):
        self.embedding_layer()
        self.bi_lstm_layer()
        self.max_pooling_layer()
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

    def multi_bi_lstm_layer(self):
        """
        实现多层的LSTM结构
        :return:
        """
        fw_hidden_layers = []
        bw_hidden_layers = []

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

                    fw_hidden_layers.append(lstm_fw_cell)
                    bw_hidden_layers.append(lstm_bw_cell)

            # 实现多层的LSTM结构， state_is_tuple=True，则状态会以元祖的形式组合(h, c)，否则列向拼接
            fw_multi_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=fw_hidden_layers, state_is_tuple=True)
            bw_multi_lstm = tf.nn.rnn_cell.MultiRNNCell(cells=bw_hidden_layers, state_is_tuple=True)

            # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
            # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],fw和bw的hidden_size一样
            # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
            outputs, self.current_state = tf.nn.bidirectional_dynamic_rnn(fw_multi_lstm, bw_multi_lstm, self.embedded_words,
                                                                          dtype=tf.float32)
            self.fw_output, self.bw_output = outputs


    def max_pooling_layer(self):
        """
        将Bi-LSTM获得的隐层输出和词向量拼接[fw_output, word_embedding, bw_output]
        将拼接后的向量非线性映射到低维
        向量中的每一个位置的值都取所有时序上的最大值，得到最终的特征向量，该过程类似于max-pool
        :return:
        """
        with tf.name_scope("context"):
            shape = [tf.shape(self.fw_output)[0], 1, tf.shape(self.fw_output)[2]]
            context_left = tf.concat([tf.zeros(shape), self.fw_output[:, :-1]], axis=1, name="context_left")
            context_right = tf.concat([self.bw_output[:, 1:], tf.zeros(shape)], axis=1, name="context_right")


        # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
        # 将Bi-LSTM获得的隐层输出和词向量拼接[fw_output, word_embedding, bw_output]
        with tf.name_scope("word_representation"):
            word_representation = tf.concat([context_left, self._embedded_words, context_right], axis=2)
            word_size = self.config.hidden_sizes[-1] * 2 + self.config.embedding_dim

        with tf.name_scope("text_representation"):
            text_w = tf.Variable(tf.random_uniform([word_size, self.config.hidden_size], -1.0, 1.0), name="text_w")
            text_b = tf.Variable(tf.constant(0.1, shape=[self.config.hidden_size]), name="text_b")

            # tf.einsum可以指定维度的消除运算
            text_representation = tf.tanh(tf.einsum('aij,jk->aik', word_representation, text_w) + text_b)

        # 做max-pool的操作，将时间步的维度消失
        self.pool_output = tf.reduce_max(text_representation, axis=1)


    def full_connection_layer(self):
        """
        全连接层，后面接dropout以及relu激活
        :return:
        """

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.pool_output, self.keep_prob)


        # 全连接层的输出
        with tf.name_scope("fully_connection_layer"):
            output_w = tf.get_variable(
                "output_w",
                shape=[self.config.hidden_size, self.config.num_labels],
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