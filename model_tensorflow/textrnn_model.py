#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-3 下午5:05
@File    : textrnn_model.py
@Desc    : 
"""

import tensorflow as tf

from model_tensorflow.basic_model import BaseModel
import configparser


class Config(object):
    """RNN配置参数"""
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
        self.rnn_gate = config.get("rnn_gate", "lstm")                     # rnn核(rnn,lstm,gru)
        self.num_layers = config.getint("num_layers", 2)                   # rnn层数
        self.hidden_size = config.getint("hidden_size", 128)               # rnn网络的隐层大小
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
        self.model_name = config.get("model_name", "textrnn")              # 模型名称



class TextRNN(BaseModel):

    def __init__(self, config, vocab_size, word_vectors):
        super(TextRNN, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        self.inputs = tf.placeholder(tf.int32, [None, config.sequence_length], name="inputs")
        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()


    def build_model(self):
        self.embedding_layer()
        # self.multi_rnn_layer()
        self.multi_bi_rnn_layer()
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


    def full_connection_layer(self):
        """
        全连接层，后面接dropout以及relu激活
        :return:
        """

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.rnn_output, self.keep_prob)


        # 全连接层的输出
        with tf.name_scope("fully_connection_layer"):
            output_w = tf.get_variable(
                "output_w",
                shape=[self._output_size, self.config.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_labels]), name="output_b")
            self.logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name="logits")
            # self.logits = tf.matmul(h_drop, output_w) + output_b
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)

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
            hiddens, states = tf.contrib.rnn.static_rnn(cell=rnn_cell, inputs=input_x1, dtype=tf.float32)
        else:
            hiddens, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=self.embedded_words, dtype=tf.float32)
            # 注意这里输出需要转置  转换为时序优先的
            # hiddens = tf.transpose(hiddens, [1, 0, 2])
            # self.rnn_output = hiddens[-1]

        self._output_size = self.config.hidden_size
        self.rnn_output = hiddens[:, -1, :]  # 取最后一个时序输出作为结果

        # # 全连接层，后面接dropout以及relu激活
        # output = tf.contrib.layers.fully_connected(inputs=hiddens[-1], num_outputs=self.label_size,
        #                                         activation_fn=tf.nn.softmax)
        # fc = tf.contrib.layers.dropout(output, self.dropout_keep_prob)
        # fc = tf.nn.relu(fc)


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
            hiddens, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell,
                                                                                         inputs=self.embedded_words,
                                                                                         dtype=tf.float32)
            # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2], 传入到下一层Bi-LSTM中
            # 按axis=2合并 (?,?,128) (?,?,128)按最后一维合并(?,28,256)
            hiddens = tf.concat(hiddens, axis=2)
            # hiddens = tf.transpose(hiddens, [1, 0, 2])
            # self.rnn_output = hiddens[-1]

        self._output_size = self.config.hidden_size * 2
        # 取出最后时间步的输出作为全连接的输入
        self.rnn_output = hiddens[:, -1, :]
        # # reshape成全连接层的输入维度
        # self.rnn_output = tf.reshape(hiddens[-1], [-1, self._output_size])

        # self.rnn_output = _outputs[:, -1, :]  # 取最后一个时序输出作为结果
        # # 全连接层，后面接dropout以及relu激活
        # output = tf.contrib.layers.fully_connected(inputs=hiddens[-1], num_outputs=self.label_size,
        #                                            activation_fn=tf.nn.softmax)
        # fc = tf.contrib.layers.dropout(output, self.dropout_keep_prob)
        # fc = tf.nn.relu(fc)


    def get_rnn_cell(self):
        if self.config.rnn_gate == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(num_units=self.config.hidden_size, state_is_tuple=True, forget_bias=1.0)
        elif self.config.rnn_gate == 'gru':
            return tf.contrib.rnn.GRUCell(num_units=self.config.hidden_size)
        else:
            return tf.contrib.rnn.BasicRNNCell(num_units=self.config.hidden_size)

    def dropout_cell(self):
        return tf.contrib.rnn.DropoutWrapper(self.get_rnn_cell(), output_keep_prob=self.keep_prob)

    def _hidden_layer(self, dropout_layer=True):

        with tf.name_scope("{}-layer".format(self.config.rnn_gate)):
            if self.config.num_layers > 1:
                cells = list()
                for i in range(self.config.num_layers):
                    if dropout_layer:
                        cells.append(self.dropout_cell())
                    else:
                        cells.append(self.get_rnn_cell())
                rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
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
                    else:
                        fw_rnn_cell.append(self.get_rnn_cell())
                for i in range(self.config.num_layers):
                    if dropout_layer:
                        bw_rnn_cell.append(self.dropout_cell())
                    else:
                        bw_rnn_cell.append(self.get_rnn_cell())
            else:
                fw_rnn_cell = self.get_rnn_cell()
                bw_rnn_cell = self.get_rnn_cell()

        return fw_rnn_cell, bw_rnn_cell




    def cal_loss(self):

        with tf.name_scope("loss"):
            # 计算交叉熵损失
            self.labels = tf.cast(self.labels, dtype=tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = tf.reduce_mean(losses)

            # self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.config.l2_reg_lambda
            # self.loss = loss + self.l2_loss
            self.loss = loss + self.config.l2_reg_lambda * self.l2_loss
