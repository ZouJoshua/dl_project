#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/31/20 1:30 PM
@File    : char_rnn_model.py
@Desc    : 

"""

import tensorflow as tf

from model_tensorflow.basic_model import BaseModel
from model_tensorflow.basic_config import ConfigBase



class Config(ConfigBase):
    """char_rnn配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)
        self.num_layers = self.config.getint("num_layers", 2)                   # lstm层数


class CharRNN(BaseModel):

    def __init__(self, config, vocab_size, word_vectors):
        super(CharRNN, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        self.initial_state = tf.placeholder(tf.float32, [None, None], name="initial_state")
        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):
        self.embedding_layer()
        self.lstm_layer()
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

    def get_a_cell(self, lstm_size, keep_prob):
        lstm = tf.nn.rnn_cell.BasicLSTMCell(lstm_size)
        drop = tf.nn.rnn_cell.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        return drop


    def lstm_layer(self):

        split_init_state = tf.split(self.initial_state, num_or_size_splits=4, axis=0)
        initial_state = (tf.nn.rnn_cell.LSTMStateTuple(h=split_init_state[0], c=split_init_state[1]),
                         tf.nn.rnn_cell.LSTMStateTuple(h=split_init_state[2], c=split_init_state[3]))

        with tf.name_scope('lstm-layer'):
            cell = tf.nn.rnn_cell.MultiRNNCell(
                [self.get_a_cell(self.config.hidden_size, self.keep_prob) for _ in range(self.config.num_layers)]
            )

            # 通过dynamic_rnn对cell展开时间维度
            outputs, final_state = tf.nn.dynamic_rnn(cell, self.embedded_words, initial_state=initial_state)

            self.final_state = tf.concat([final_state[0].h,
                                          final_state[0].c,
                                          final_state[1].h,
                                          final_state[1].c],
                                         axis=0,
                                         name="final_state")

            # 通过lstm_outputs得到概率
            seq_output = tf.concat(outputs, 1)
            self.lstm_output = tf.reshape(seq_output, [-1, self.config.hidden_size])

    def full_connection_layer(self):
        """
        全连接层，后面接dropout以及relu激活
        :return:
        """

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.lstm_output, self.keep_prob)


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