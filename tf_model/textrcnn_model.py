#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-3 下午5:05
@File    : textrnn_model.py
@Desc    : 
"""

import tensorflow as tf


class TextRCNN(object):
    """
    Using LSTM or GRU neural network for text classification
    """
    def __init__(self,
                 sentence_len,
                 label_size,
                 batch_size,
                 hidden_unit,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 vocab_size,
                 embed_size,
                 is_training,
                 cell='lstm',
                 clip_gradients=5.0):
        self.sentence_len = sentence_len
        self.label_size = label_size
        self.batch_size = batch_size

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_dim = hidden_unit
        self.is_training = is_training
        self.learning_rate = learning_rate

        self.is_training_flag = is_training
        self.learning_rate = learning_rate
        self.decay_rate = learning_decay_rate
        self.decay_steps = learning_decay_steps
        self.gate = cell

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
        fully_connection_and_softmax_layer
        """
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.sentence)  # [None,sencente_len,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)  # [None,sencente_len,embed_size,1]

        print("use rcnn layer")
        h = self.rcnn_layer(self.embedded_words, self.sentence_len, self.hidden_dim, n_hidden_layer=1, if_dropout=True, static=False)

        # 5. logits(use linear layer) and predictions(argmax)
        # full coneection and softmax output
        with tf.variable_scope('fully_connection_layer'):
            # shape:[None, self.label_size]==tf.matmul([None,self.hidden_dim],[self.hidden_dim, self.label_size])
            # logits = tf.nn.softmax(tf.matmul(h, self.w) + self.b, name='logits')
            logits = tf.matmul(h, self.w) + self.b
        return logits

    def get_rnn_cell(self, hidden_unit):
        if self.gate == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(num_units=hidden_unit, forget_bias=1.0)
        elif self.gate == 'gru':
            return tf.contrib.rnn.GRUCell(num_units=hidden_unit)
        else:
            return tf.contrib.rnn.BasicRNNCell(num_units=hidden_unit)

    def dropout_cell(self, input_cell):
        return tf.contrib.rnn.DropoutWrapper(input_cell, output_keep_prob=self.dropout_keep_prob)

    def hidden_layer(self, hidden_unit, dropout_layer=False, multi_layer=1):

        if multi_layer > 1:
            cells = list()
            for i in range(multi_layer):
                if dropout_layer:
                    cells.append(self.dropout_cell(self.get_rnn_cell(hidden_unit)))
                else:
                    cells.append(self.get_rnn_cell(hidden_unit))
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        else:
            rnn_cell = self.get_rnn_cell(hidden_unit)

        return rnn_cell

    def hidden_bi_lstm_layer(self, hidden_unit, dropout_layer=False, multi_layer=1):

        with tf.name_scope("bi-lstm"):

            if multi_layer > 1:
                fw_rnn_cell = list()
                bw_rnn_cell = list()
                for i in range(multi_layer):
                    if dropout_layer:
                        fw_rnn_cell.append(self.dropout_cell(self.get_rnn_cell(hidden_unit)))
                    else:
                        fw_rnn_cell.append(self.get_rnn_cell(hidden_unit))
                for i in range(multi_layer):
                    if dropout_layer:
                        bw_rnn_cell.append(self.dropout_cell(self.get_rnn_cell(hidden_unit)))
                    else:
                        bw_rnn_cell.append(self.get_rnn_cell(hidden_unit))
            else:
                fw_rnn_cell = self.get_rnn_cell(hidden_unit)
                bw_rnn_cell = self.get_rnn_cell(hidden_unit)

        return fw_rnn_cell, bw_rnn_cell


    def rcnn_layer(self, input_x, n_steps, n_hidden_unit, n_hidden_layer=1, if_dropout=False, static=True):
        """
        1，利用Bi-LSTM获得上下文的信息
        2，将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput;wordEmbedding;bwOutput]
        3，将2所得的词表示映射到低维
        4，hidden_size上每个位置的值都取时间步上最大的值，类似于max-pool
        :param input_x: 输入数据
        :param n_steps: 时序
        :param n_hidden_unit: 隐藏层神经元个数
        :param n_hidden_layer: 隐藏层层数
        :param if_dropout: 是否用dropout
        :param static: 是否用动态计算
        :return:
        """
        fw_rnn_cell, bw_rnn_cell = self.hidden_bi_lstm_layer(n_hidden_unit, dropout_layer=if_dropout, multi_layer=n_hidden_layer)
        (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_rnn_cell, cell_bw=bw_rnn_cell, inputs=input_x,
                                                         dtype=tf.float32)

        with tf.name_scope("context"):
            shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
            c_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="context_left")
            c_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word_representation"):
            y2 = tf.concat([c_left, input_x, c_right], axis=2, name="word_representation")
            embedding_size = 2 * n_hidden_unit + self.embed_size

        # max_pooling层
        with tf.name_scope("max_pooling"):
            fc = tf.layers.dense(y2, self.hidden_dim, activation=tf.nn.relu, name='fc1')
            fc_pool = tf.reduce_max(fc, axis=1)

        return fc_pool


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