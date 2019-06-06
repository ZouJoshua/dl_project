#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-3 下午5:05
@File    : textrnn_model.py
@Desc    : 
"""

import tensorflow as tf

from tensorflow.contrib import rnn


class TextRNN(object):
    def __init__(self,
                 filter_sizes,
                 num_filters,
                 label_size,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 batch_size,
                 sentence_len,
                 vocab_size,
                 embed_size,
                 is_training,
                 clip_gradients=5.0):
        self.label_size = label_size
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training_flag = is_training
        self.learning_rate = learning_rate
        self.decay_rate = learning_decay_rate
        self.decay_steps = learning_decay_steps

        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters

        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
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
        self.w = tf.get_variable("w", shape=[self.num_filters_total, self.label_size], initializer=self.initializer)  # [embed_size,label_size], w是随机初始化来的
        self.b = tf.get_variable("b", shape=[self.label_size])       # [label_size]

    def inference(self):
        """
        embedding layers
        convolutional layer
        max-pooling
        softmax layer"""
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.sentence)  # [None,sencente_len,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)  # [None,sencente_len,embed_size,1]

        # if self.use_mulitple_layer_cnn: # this may take 50G memory.
        #    print("use multiple layer CNN")
        #    h=self.cnn_multiple_layers()
        # else: # this take small memory, less than 2G memory
        print("use single layer CNN")
        h = self.cnn_single_layer()
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.variable_scope('fully_connection_layer'):
            logits = tf.matmul(h, self.w) + self.b  # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def single_layer_static_lstm(self, input_x, n_steps, n_hidden):
        '''
        返回静态单层LSTM单元的输出，以及cell状态

        args:
            input_x:输入张量 形状为[batch_size,n_steps,n_input]
            n_steps:时序总数
            n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
        '''

        # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
        # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
        input_x1 = tf.unstack(input_x, num=n_steps, axis=1)

        # 可以看做隐藏层
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0)
        # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
        hiddens, states = tf.contrib.rnn.static_rnn(cell=lstm_cell, inputs=input_x1, dtype=tf.float32)

        return hiddens, states

    def single_layer_static_gru(self, input_x, n_steps, n_hidden):
        '''
        返回静态单层GRU单元的输出，以及cell状态

        args:
            input_x:输入张量 形状为[batch_size,n_steps,n_input]
            n_steps:时序总数
            n_hidden：gru单元输出的节点个数 即隐藏层节点数
        '''

        # 把输入input_x按列拆分，并返回一个有n_steps个张量组成的list 如batch_sizex28x28的输入拆成[(batch_size,28),((batch_size,28))....]
        # 如果是调用的是静态rnn函数，需要这一步处理   即相当于把序列作为第一维度
        input_x1 = tf.unstack(input_x, num=n_steps, axis=1)

        # 可以看做隐藏层
        gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
        # 静态rnn函数传入的是一个张量list  每一个元素都是一个(batch_size,n_input)大小的张量
        hiddens, states = tf.contrib.rnn.static_rnn(cell=gru_cell, inputs=input_x1, dtype=tf.float32)

        return hiddens, states

    def single_layer_dynamic_lstm(self, input_x, n_steps, n_hidden):
        '''
        返回动态单层LSTM单元的输出，以及cell状态

        args:
            input_x:输入张量  形状为[batch_size,n_steps,n_input]
            n_steps:时序总数
            n_hidden：LSTM单元输出的节点个数 即隐藏层节点数
        '''
        # 可以看做隐藏层
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(num_units=n_hidden, forget_bias=1.0)
        # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
        hiddens, states = tf.nn.dynamic_rnn(cell=lstm_cell, inputs=input_x, dtype=tf.float32)

        # 注意这里输出需要转置  转换为时序优先的
        hiddens = tf.transpose(hiddens, [1, 0, 2])
        return hiddens, states

    def single_layer_dynamic_gru(self, input_x, n_steps, n_hidden):
        '''
        返回动态单层GRU单元的输出，以及cell状态

        args:
            input_x:输入张量 形状为[batch_size,n_steps,n_input]
            n_steps:时序总数
            n_hidden：gru单元输出的节点个数 即隐藏层节点数
        '''

        # 可以看做隐藏层
        gru_cell = tf.contrib.rnn.GRUCell(num_units=n_hidden)
        # 动态rnn函数传入的是一个三维张量，[batch_size,n_steps,n_input]  输出也是这种形状
        hiddens, states = tf.nn.dynamic_rnn(cell=gru_cell, inputs=input_x, dtype=tf.float32)

        # 注意这里输出需要转置  转换为时序优先的
        hiddens = tf.transpose(hiddens, [1, 0, 2])
        return hiddens, states


    def cnn_multiple_layers(self):
        # loop each filter size
        # for each filter, do: convolution-pooling layer, feature shape is 4-d. Feature is a new variable
        # step1.create filters
        # step2.conv (CNN->BN->relu)
        # step3.apply nolinearity(tf.nn.relu)
        # step4.max-pooling(tf.nn.max_pool)
        # step5.dropout
        pooled_outputs = []
        print("sentence_embeddings_expanded:", self.sentence_embeddings_expanded)
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('cnn_multiple_layers' + "convolution-pooling-%s" % filter_size):
                # Layer1:CONV-RELU
                # 1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="SAME", name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn1')
                print(i, "conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 2) CNN->BN->relu
                h = tf.reshape(h, [-1, self.sentence_len, self.num_filters, 1])  # shape:[batch_size,sequence_length,num_filters,1]
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size, [filter_size, self.num_filters, 1, self.num_filters], initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2), "relu2")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 3. Max-pooling
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, self.sentence_len, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling_max)
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length,1,num_filters]
        # concat
        h = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_filters*len(self.filter_sizes)]
        print("h.concat:", h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_len - filter_size + 1,num_filters]
        return h  # [batch_size,sequence_len - filter_size + 1,num_filters]

    # def loss_multilabel(self, l2_lambda=0.0001):  # 0.0001
    #     with tf.name_scope("loss"):
    #         # let `x = logits`, `z = labels`.
    #         # The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    #         losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
    #         #losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
    #         #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
    #         print("sigmoid_cross_entropy_with_logits.losses:", losses)
    #         losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
    #         loss = tf.reduce_mean(losses)         # shape=().   average loss in the batch
    #         l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
    #         loss = loss+l2_losses
    #     return loss


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