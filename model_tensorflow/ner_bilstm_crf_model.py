#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/12/19 8:06 PM
@File    : ner_bilstm_crf_model.py
@Desc    : 

"""

import tensorflow as tf
from model_tensorflow.basic_model import BaseModel
from model_tensorflow.basic_config import ConfigBase


class Config(ConfigBase):
    """ner_bilstm_crf配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)
        self.rnn_gate = self.config.get("rnn_gate", "lstm")  # rnn核(rnn,lstm,gru)
        self.num_layers = self.config.getint("num_layers", 1)  # rnn层数



class NERTagger(BaseModel):

    def __init__(self, config, vocab_size, word_vectors):
        super(NERTagger, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")  # 数据输入
        self.labels = tf.placeholder(tf.int32, [None, self.config.sequence_length], name="labels")  # 标签
        self.sequence_lengths = tf.placeholder(dtype=tf.int32, shape=[None], name='sequence_lengths_vector')
        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):
        self.embedding_layer()
        self.multi_bi_rnn_layer()
        self.crf_layer()
        self.cal_loss()

        self.predictions = self.get_predictions()
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()

    def embedding_layer(self):
        """
        词嵌入层
        :return:
        """
        with tf.device("/cpu:0"):
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

    def multi_rnn_layer(self, static=False):
        """
        多层单向rnn网络
        :param static: 是否用动态计算
        :return:
        """

        rnn_cell = self._hidden_layer()

        if static:
            # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
            input_x1 = tf.unstack(self.embedded_words, num=self.config.sequence_length, axis=1)
            hiddens, states = tf.nn.static_rnn(cell=rnn_cell, inputs=input_x1, dtype=tf.float32)
        else:
            hiddens, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embedded_words, dtype=tf.float32)
            # 注意这里输出需要转置  转换为时序优先的
            # hiddens = tf.transpose(hiddens, [1, 0, 2])
            # self.rnn_output = hiddens[-1]

        self._output_size = self.config.hidden_size
        self.rnn_output = hiddens[:, -1, :]  # 取最后一个时序输出作为结果


    def multi_bi_rnn_layer(self, static=False):
        """
        多层双向rnn网络（默认bi-lstm）
        :param static: 是否用动态计算
        :return:
        """
        fw_rnn_cell, bw_rnn_cell = self._hidden_bi_layer()

        if static:
            # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
            input_x1 = tf.unstack(self.embedded_words, num=self.config.sequence_length, axis=1)
            hiddens, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_rnn(fw_rnn_cell, bw_rnn_cell,
                                                                                 inputs=input_x1, dtype=tf.float32)
        else:
            # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
            # hiddens, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell,
            #                                                                              inputs=self.embedded_words,
            #                                                                              dtype=tf.float32)
            hiddens, fw_state, bw_state = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell,
                                                                     self.embedded_words, dtype=tf.float32)


            # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
            # 按axis=2合并 (?,?,128) (?,?,128)按最后一维合并(?,28,256)
            hiddens = tf.concat(hiddens, axis=2, name="bi_lstm_concat")

        self.rnn_output = hiddens
        self.rnn_output_shape = hiddens.get_shape()


    def crf_layer(self):
        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.rnn_output, self.keep_prob)
        self._output_size = self.config.hidden_size * 2

        # Linear-Chain CRF Layer
        with tf.name_scope("crf_connection_layer"):
            crf_input = tf.reshape(h_drop, shape=[-1, self._output_size], name='contact')
            crf_w = tf.get_variable('crf_weights',
                                    shape=[self._output_size, self.config.num_labels],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32, trainable=True)
            crf_b = tf.get_variable('crf_b', initializer=tf.zeros(shape=[self.config.num_labels]), name="crf_b")
            p = tf.nn.relu(tf.matmul(crf_input, crf_w) + crf_b)
            # CRF x input, shape [batch_size, num_steps, label_num]
            self.logit = tf.reshape(p, shape=[-1, self.rnn_output_shape[1], self.config.num_labels], name='logits')
            self.l2_loss += tf.nn.l2_loss(crf_w)
            self.l2_loss += tf.nn.l2_loss(crf_b)


    def get_rnn_cell(self):
        """
        自定义返回rnn单元(lstm\rnn\gru)
        :return:
        """
        if self.config.rnn_gate == 'lstm':
            return tf.nn.rnn_cell.BasicLSTMCell(num_units=self.config.hidden_size, state_is_tuple=True, forget_bias=1.0)
        elif self.config.rnn_gate == 'gru':
            return tf.nn.rnn_cell.GRUCell(num_units=self.config.hidden_size)
        else:
            return tf.nn.rnn_cell.BasicRNNCell(num_units=self.config.hidden_size)

    def dropout_cell(self):
        """
        添加dropout层
        :return:
        """
        return tf.nn.rnn_cell.DropoutWrapper(self.get_rnn_cell(), output_keep_prob=self.keep_prob)

    def _hidden_layer(self, dropout_layer=True):

        with tf.name_scope("{}-layer".format(self.config.rnn_gate)):
            if self.config.num_layers > 1:
                cells = list()
                for i in range(self.config.num_layers):
                    if dropout_layer:
                        cells.append(self.dropout_cell())
                    else:
                        cells.append(self.get_rnn_cell())
                rnn_cell = tf.nn.rnn_cell.MultiRNNCell(cells, state_is_tuple=True)
            else:
                rnn_cell = self.get_rnn_cell()

        return rnn_cell

    def _hidden_bi_layer(self, dropout_layer=True):

        with tf.name_scope("bi-{}-layer".format(self.config.rnn_gate)):

            if self.config.num_layers > 1:
                fw_rnn_cell = list()
                bw_rnn_cell = list()
                for i in range(self.config.num_layers):
                    if dropout_layer:
                        fw_rnn_cell.append(self.dropout_cell())
                        bw_rnn_cell.append(self.dropout_cell())
                    else:
                        fw_rnn_cell.append(self.get_rnn_cell())
                        bw_rnn_cell.append(self.get_rnn_cell())

            else:
                fw_rnn_cell = self.get_rnn_cell()
                bw_rnn_cell = self.get_rnn_cell()

        return fw_rnn_cell, bw_rnn_cell


    def cal_loss(self):
        """
        计算crf损失
        :return:
        """
        with tf.name_scope("crf_loss"):
            log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(self.logit, self.labels, sequence_lengths=self.sequence_lengths)
            loss = -tf.reduce_mean(log_likelihood)
            self.loss = loss + self.l2_loss

