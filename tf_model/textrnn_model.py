#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-3 下午5:05
@File    : textrnn_model.py
@Desc    : 
"""

import tensorflow as tf


class TextRNN(object):
    def __init__(self,
                 sentence_len,
                 label_size,
                 batch_size,
                 hidden_dim,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 vocab_size,
                 embed_size,
                 is_training,
                 clip_gradients=5.0):
        self.sentence_len = sentence_len
        self.label_size = label_size
        self.batch_size = batch_size

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_dim = hidden_dim
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.num_sampled = 20

        self.is_training_flag = is_training
        self.learning_rate = learning_rate
        self.decay_rate = learning_decay_rate
        self.decay_steps = learning_decay_steps
        self.gate = "lstm"

        self.clip_gradients = clip_gradients

        self.build_graph()

    def add_placeholders(self):
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")  # X
        self.label = tf.placeholder(tf.int32, [None, ], name="label")  # y:[None,label_size]
        # self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.label_size], name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


    def init_weights(self):
        """define all weights here"""
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        with tf.name_scope("embedding_layer"):
            self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
        self.w = tf.get_variable("w", shape=[self.hidden_dim, self.label_size], initializer=self.initializer)  # [embed_size,label_size], w是随机初始化来的
        self.b = tf.get_variable("b", shape=[self.label_size])       # [label_size]

    def inference(self):
        """
        embedding layers
        single_hidden_layer
        """
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.sentence)  # [None,sencente_len,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)  # [None,sencente_len,embed_size,1]

        print("use single layer RNN")
        h = self.rnn_single_layer(self.embedded_words, self.sentence_len, self.hidden_dim, static=True)
        # print("use single layer bi-RNN")
        # h = self.rnn_single_bi_layer(self.embedded_words, self.sentence_len, self.hidden_dim, static=True)
        # print("use multi layer RNN")
        # h = self.rnn_multi_layer(self.embedded_words, self.sentence_len, self.hidden_dim, static=True)
        # print("use multi layer bi-RNN")
        # h = self.rnn_multi_bi_layer(self.embedded_words, self.sentence_len, self.hidden_dim, static=True)

        # 5. logits(use linear layer) and predictions(argmax)
        with tf.variable_scope('fully_connection_layer'):
            # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
            logits = tf.matmul(h, self.w) + self.b
        return logits

    def hidden_layer(self, hidden_dim, dropout_layer=False, multi_layer=1):
        if self.gate == 'lstm':
            hidden_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=1.0)
        elif self.gate == 'gru':
            hidden_cell = tf.contrib.rnn.GRUCell(num_units=hidden_dim)
        else:
            hidden_cell = tf.contrib.rnn.BasicRNNCell(num_units=hidden_dim)

        if dropout_layer:
            dropout_cell = tf.contrib.rnn.DropoutWrapper(hidden_cell, output_keep_prob=self.dropout_keep_prob)
        else:
            dropout_cell = hidden_cell

        if multi_layer > 1:
            cells = [dropout_cell for _ in range(multi_layer)]
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        else:
            rnn_cell = dropout_cell

        return rnn_cell

    def hidden_bi_lstm_layer(self, hidden_dim, dropout_layer=False, multi_layer=1):

        fw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=1.0)
        bw_cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, forget_bias=1.0)

        if dropout_layer:
            fw_dropout_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=self.dropout_keep_prob)
            bw_dropout_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=self.dropout_keep_prob)
        else:
            fw_dropout_cell = fw_cell
            bw_dropout_cell = bw_cell

        if multi_layer != 1:
            fw_rnn_cell = [fw_dropout_cell for _ in range(multi_layer)]
            bw_rnn_cell = [bw_dropout_cell for _ in range(multi_layer)]
        else:
            fw_rnn_cell = fw_dropout_cell
            bw_rnn_cell = bw_dropout_cell

        return fw_rnn_cell, bw_rnn_cell


    def rnn_single_layer(self, input_x, n_steps, n_hidden, static=True):
        """
        单层rnn单向网络
        :param input_x:
        :param n_steps:
        :param n_hidden:
        :param static:
        :return:
        """
        rnn_cell = self.hidden_layer(n_hidden, dropout_layer=False, multi_layer=1)
        if static:
            # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
            input_x1 = tf.unstack(input_x, num=n_steps, axis=1)
            hiddens, states = tf.contrib.rnn.static_rnn(cell=rnn_cell, inputs=input_x1, dtype=tf.float32)
        else:
            # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
            hiddens, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=input_x, dtype=tf.float32)
            # 注意这里输出需要转置  转换为时序优先的
            hiddens = tf.transpose(hiddens, [1, 0, 2])
        print(type(hiddens))
        print(hiddens[-1])
        # 全连接层，后面接dropout以及relu激活
        # output = tf.contrib.layers.fully_connected(inputs=hiddens[-1], num_outputs=self.label_size,
        #                                            activation_fn=tf.nn.softmax)
        # print(type(output))
        # print(output)
        # fc = tf.contrib.layers.dropout(output, self.dropout_keep_prob)
        # fc = tf.nn.relu(fc)

        return hiddens[-1]

    def rnn_multi_layer(self, input_x, n_steps, n_hidden, static=True):
        rnn_cell = self.hidden_layer(n_hidden, dropout_layer=False, multi_layer=2)

        if static:
            # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
            input_x1 = tf.unstack(input_x, num=n_steps, axis=1)
            hiddens, states = tf.contrib.rnn.static_rnn(cell=rnn_cell, inputs=input_x1, dtype=tf.float32)
        else:
            hiddens, states = tf.nn.dynamic_rnn(cell=rnn_cell, inputs=input_x, dtype=tf.float32)
            hiddens = tf.transpose(hiddens, [1, 0, 2])
        # 全连接层，后面接dropout以及relu激活
        output = tf.contrib.layers.fully_connected(inputs=hiddens[-1], num_outputs=self.label_size,
                                                activation_fn=tf.nn.softmax)
        fc = tf.contrib.layers.dropout(output, self.dropout_keep_prob)
        fc = tf.nn.relu(fc)

        return fc

    def rnn_single_bi_layer(self, input_x, n_steps, n_hidden, static=True):
        fw_rnn_cell, bw_rnn_cell = self.hidden_bi_lstm_layer(n_hidden, dropout_layer=False, multi_layer=1)
        if static:
            # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
            input_x1 = tf.unstack(input_x, num=n_steps, axis=1)
            hiddens, fw_state, bw_state = tf.contrib.rnn.static_bidirectional_rnn(cell_fw=fw_rnn_cell,
                                                                                  cell_bw=bw_rnn_cell, inputs=input_x1,
                                                                                  dtype=tf.float32)
        else:
            hiddens, state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_rnn_cell, cell_bw=bw_rnn_cell, inputs=input_x,
                                                             dtype=tf.float32)
            # 按axis=2合并 (?,28,128) (?,28,128)按最后一维合并(?,28,256)
            hiddens = tf.concat(hiddens, axis=2)
            hiddens = tf.transpose(hiddens, [1, 0, 2])
        # 全连接层，后面接dropout以及relu激活
        output = tf.contrib.layers.fully_connected(inputs=hiddens[-1], num_outputs=self.label_size,
                                                activation_fn=tf.nn.softmax)
        fc = tf.contrib.layers.dropout(output, self.dropout_keep_prob)
        fc = tf.nn.relu(fc)
        return fc

    def rnn_multi_bi_layer(self, input_x, n_steps, n_hidden, static=True):
        fw_rnn_cell, bw_rnn_cell = self.hidden_bi_lstm_layer(n_hidden, dropout_layer=False, multi_layer=2)

        if static:
            # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
            input_x1 = tf.unstack(input_x, num=n_steps, axis=1)
            hiddens, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_rnn(fw_rnn_cell, bw_rnn_cell,
                                                                             inputs=input_x1, dtype=tf.float32)
        else:
            hiddens, fw_state, bw_state = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(fw_rnn_cell, bw_rnn_cell,
                                                                                         inputs=input_x,
                                                                                         dtype=tf.float32)
            hiddens = tf.concat(hiddens, axis=2)
            hiddens = tf.transpose(hiddens, [1, 0, 2])
        # 全连接层，后面接dropout以及relu激活
        output = tf.contrib.layers.fully_connected(inputs=hiddens[-1], num_outputs=self.label_size,
                                                   activation_fn=tf.nn.softmax)
        fc = tf.contrib.layers.dropout(output, self.dropout_keep_prob)
        fc = tf.nn.relu(fc)

        return fc


    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            self.y_true = tf.one_hot(self.label, self.label_size)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            # sigmoid_cross_entropy_with_logits.
            # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def acc(self):
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
        self.y_pred = tf.one_hot(self.predictions, self.label_size)
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        return accuracy

    def train_old(self):
        """based on the loss, use SGD to update parameter"""
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op

    def train(self):
        """based on the loss, use SGD to update parameter"""
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_= learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op

    def build_graph(self):
        self.add_placeholders()
        self.init_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.accuracy = self.acc()