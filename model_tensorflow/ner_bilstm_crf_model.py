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
import numpy as np

class Config(ConfigBase):
    """ner配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)

        self.tag_scheme = self.config.get("tag_scheme", "BIO")          # 序列标注编码格式BIO, BIOES
        self.seg_embedding = self.config.getboolean("seg_embedding", True)      # 是否使用分词特征
        self.segment_size = self.config.getint("segment_size")                      # 分词特征数
        self.char_embedding_dim = self.config.getint("char_embedding_dim", 300)  # 单字维度特征
        self.seg_embedding_dim = self.config.getint("seg_embedding_dim", 300)    # 分词维度特征
        self.embedding_dim = self.char_embedding_dim + self.seg_embedding_dim

        # bilstm 模型参数
        self.rnn_gate = self.config.get("rnn_gate", "lstm")  # rnn核(rnn,lstm,gru)
        self.num_layers = self.config.getint("num_layers", 1)  # rnn层数
        # idcnn 模型参数
        self.repeat_times = self.config.getint("repeat_times", 3)                    #空洞卷积网络重复次数
        self.dilation_sizes = eval(self.config.get("dilation_sizes", "[1,1,2]"))       # 空洞卷积核尺寸, a list of int. e.g. [1,1,2]
        self.filter_width = self.config.getint("filter_width", 3)                      # 卷积宽度
        self.num_filter = self.config.getint("num_filter")                              # 卷积核数量(channels数)
        self.is_training = self.config.getboolean("is_training", True)                  # 是否训练



class NERTagger(BaseModel):

    def __init__(self, config, vocab_size, word_vectors):
        super(NERTagger, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        self.inputs = tf.placeholder(tf.int32, [None, None], name="inputs")  # 数据输入
        self.seg_inputs = tf.placeholder(tf.int32, [None, None], name="segments")
        self.labels = tf.placeholder(tf.int32, [None, None], name="labels")  # 标签

        used = tf.sign(tf.abs(self.inputs))
        length = tf.reduce_sum(used, reduction_indices=1)
        self.lengths = tf.cast(length, tf.int32)  # 计算真实序列真实长度
        self.num_setps = tf.shape(self.inputs)[-1]
        self.best_test_f1 = tf.Variable(0.0, trainable=False)

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):
        self.embedding_layer()
        self.multi_bi_rnn_layer()
        # self.idcnn_layer()
        self.full_connection_layer()
        self.cal_loss()

        # self.predictions = self.get_predictions()
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()

    def embedding_layer(self):
        """
        :param word_inputs: one-hot编码
        :param seg_inputs: 分词特征
        :return:
        """
        with tf.device("/cpu:0"):
            with tf.variable_scope("embedding-layer"):
                # 利用预训练的词向量初始化词嵌入矩阵
                if self.word_vectors is not None:
                    char_embedding = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                              name="char_embedding")
                else:
                    char_embedding = tf.get_variable("char_embedding", shape=[self.vocab_size, self.config.char_embedding_dim],
                                                  initializer=tf.contrib.layers.xavier_initializer())
                # 利用词嵌入矩阵将输入的数据中的词转换成词向量，
                # 维度[batch_size, sequence_length, embedding_dim]
                self.embedded_char = tf.nn.embedding_lookup(char_embedding, self.inputs)

                if self.config.seg_embedding:
                    with tf.variable_scope("seg_embedding"):
                        seg_embedding = tf.get_variable(name="seg_embedding", shape=[self.config.segment_size, self.config.seg_embedding_dim],
                                                        initializer=tf.contrib.layers.xavier_initializer())
                        self.embedded_seg = tf.nn.embedding_lookup(seg_embedding, self.seg_inputs)

                # embedding层单词和分词信息
                self.embedded_words = tf.concat([self.embedded_char, self.embedded_seg], axis=-1)


    def idcnn_layer(self):
        """
        idcnn_inputs: [batch_size, num_steps, emb_size]
        :return: [batch_size, num_steps, total_width_for_last_dim]
        """
        idcnn_inputs = tf.expand_dims(self.embedded_words, 1)
        reuse = False
        if not self.config.is_training:
            reuse = True
        with tf.variable_scope('idcnn'):
            filter_shape = [1, self.config.filter_width, self.config.embedding_dim, self.config.num_filters]

            filter_weights = tf.get_variable(
                "idcnn_filter",
                shape=filter_shape,
                initializer=tf.contrib.layers.xavier_initializer())

            layer_input = tf.nn.conv2d(
                idcnn_inputs,
                filter_weights,
                strides=[1, 1, 1, 1],
                padding='SAME',
                name='init_layer'
            )

            final_out_from_layers = []
            total_width_for_last_dim = 0
            for j in range(self.config.repeat_times):
                for i in range(self.config.dilation_sizes):

                    is_last = True if i == (len(self.config.dilation_sizes) - 1) else False
                    with tf.variable_scope('conv-layer-%d' % i, reuse=tf.AUTO_REUSE):
                        w = tf.get_variable(
                            'fliter_w',
                            shape=[1, self.config.filter_width, self.config.num_filter, self.config.num_filter],
                            initializer=tf.contrib.layers.xavier_initializer())

                        b = tf.get_variable('filter_b', shape=[self.config.num_filter])

                        conv = tf.nn.atrous_conv2d(
                            layer_input,
                            w,
                            rate=self.config.dilation_sizes[i],
                            padding="SAME"
                        )

                        conv = tf.nn.bias_add(conv, b)

                        conv = tf.nn.relu(conv)

                        if is_last:
                            final_out_from_layers.append(conv)
                            total_width_for_last_dim += self.config.num_filter
                        layer_input = conv

            final_out = tf.concat(axis=3, values=final_out_from_layers)
            keepProb = 1.0 if reuse else 0.5
            final_out = tf.nn.dropout(final_out, keepProb)

            final_out = tf.squeeze(final_out, [1])
            self._output_size = total_width_for_last_dim
            self.nn_output = tf.reshape(final_out, [-1, total_width_for_last_dim])

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

        self._output_size = self.config.hidden_size * 2
        self.nn_output = tf.reshape(hiddens, shape=[-1, self._output_size], name='contact')


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
            hiddens, state = tf.nn.bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell,
                                                                     self.embedded_words, dtype=tf.float32)


            # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
            # 按axis=2合并 (?,?,128) (?,?,128)按最后一维合并(?,28,256)
            hiddens = tf.concat(hiddens, axis=2, name="bi_lstm_concat")


        self._output_size = self.config.hidden_size * 2
        self.nn_output = tf.reshape(hiddens, shape=[-1, self._output_size], name='contact')


    def full_connection_layer(self):
        """
        全连接层
        :return:
        """
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.nn_output, self.keep_prob)

        # Linear-Chain CRF Layer
        with tf.name_scope("project-connection-layer"):
            fc_w = tf.get_variable('fc_weights',
                                    shape=[self._output_size, self.config.num_labels],
                                    initializer=tf.contrib.layers.xavier_initializer(),
                                    dtype=tf.float32, trainable=True)
            fc_b = tf.get_variable('fc_b', initializer=tf.zeros(shape=[self.config.num_labels]))
            pred = tf.matmul(h_drop, fc_w) + fc_b
            self.logits = tf.reshape(pred, [-1, self.num_setps, self.config.num_labels], name='logits')
            self.l2_loss += tf.nn.l2_loss(fc_w)
            self.l2_loss += tf.nn.l2_loss(fc_b)



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
            small_value = -10000.0
            # 第一个时刻加一个时刻
            start_logits = tf.concat(
                [small_value * tf.ones(shape=[self.config.batch_size, 1, self.config.num_labels]),
                tf.zeros(shape=[self.config.batch_size, 1, 1])], axis=-1)
            # 每一个时刻加一个状态
            pad_logits = tf.cast(
                small_value * tf.ones(shape=[self.config.batch_size, self.num_setps, 1]),
                dtype=tf.float32)

            logits = tf.concat([self.logits, pad_logits], axis=-1)
            logits = tf.concat([start_logits, logits], axis=1)

            targets = tf.concat(
                [tf.cast(self.config.num_labels * tf.ones([self.config.batch_size, 1]), tf.int32), self.labels], axis=-1)

            # self.config.num_labels+1表示转移概率矩阵融入初始概率
            self.transition_matrix = tf.get_variable("transitions", shape=[self.config.num_labels+1, self.config.num_labels+1],
                initializer=tf.truncated_normal_initializer())

            log_likelihood, self.transition_matrix = tf.contrib.crf.crf_log_likelihood(
                logits,
                tag_indices=targets,
                transition_params=self.transition_matrix,
                sequence_lengths=self.lengths)
            loss = -tf.reduce_mean(log_likelihood)
            self.loss = loss + self.l2_loss



    def train(self, sess, batch, dropout_prob):
        """
        训练模型
        :param sess: tf的会话对象
        :param batch: batch数据
        :param dropout_prob: dropout比例
        :return: 损失和预测结果
        """

        feed_dict = {self.inputs: batch["char_input"],
                     self.seg_inputs: batch["seg_input"],
                     self.labels: batch["y"],
                     self.keep_prob: dropout_prob}

        # 训练模型
        _, summary, step, loss, logits, lengths, transition_matrix = sess.run(
            [self.train_op, self.summary_op, self.global_step, self.loss, self.logits, self.lengths, self.transition_matrix],
                                                 feed_dict=feed_dict)
        predictions = self.decode(logits, lengths, transition_matrix)

        return summary, step, loss, predictions

    def get_predictions(self):

        predictions = self.decode(self.logits, self.lengths, self.transition_matrix)
        return predictions


    def eval(self, sess, batch):
        """
        验证模型
        :param sess: tf中的会话对象
        :param batch: batch数据
        :return: 损失和预测结果
        """
        feed_dict = {self.inputs: batch["char_input"],
                     self.seg_inputs: batch["seg_input"],
                     self.labels: batch["y"],
                     self.keep_prob: 1.0}

        summary, step, loss, logits, lengths, transition_matrix = sess.run(
            [self.summary_op, self.global_step, self.loss, self.logits, self.lengths, self.transition_matrix], feed_dict=feed_dict)
        predictions = self.decode(logits, lengths, transition_matrix)

        return summary, step, loss, lengths, transition_matrix, predictions


    def decode(self, logits, lengths, matrix):
        """
        :param logits: [batch_size,num_steps, num_tags
        :param lengths:
        :param matrix:
        :return:
        """
        paths = []
        small = -1000.0
        start = np.asarray([[small]*self.config.num_labels + [0]])
        for score, length in zip(logits, lengths):
            score = score[:length]
            pad = small * np.ones([length, 1])
            logits = np.concatenate([score, pad], axis=1)
            logits = np.concatenate([start, logits], axis=0)
            path, _ = tf.contrib.crf.viterbi_decode(logits, matrix)

            paths.append(path[1:])
        return paths