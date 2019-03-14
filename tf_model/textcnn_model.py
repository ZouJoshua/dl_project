#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:33
@File    : textcnn_model.py
@Desc    : TextCNN:
            1. embeddding layers
            2.convolutional layer
            3.max-pooling
            4.softmax layer
"""


import tensorflow as tf


class TextCNN:
    def __init__(self, filter_sizes, num_filters, label_size, learning_rate, batch_size, decay_steps, decay_rate,
                 title_length, vocab_size, embed_size, is_training,
                 initializer=tf.random_normal_initializer(stddev=0.1), clip_gradients=5.0):
        self.label_size = label_size
        self.batch_size = batch_size
        self.title_length = title_length
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps

        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = initializer
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.clip_gradients = clip_gradients

        self.title = tf.placeholder(tf.int32, [None, self.title_length], name="sentence")  # X
        self.label = tf.placeholder(tf.int32, [None, ], name="label")  # y:[None,label_size]
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="global_Step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_Step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.init_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.label)
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="Accuracy")

    def init_weights(self):
        with tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.w = tf.get_variable("w", shape=[self.num_filters_total, self.label_size], initializer=self.initializer)  # [embed_size,label_size], w是随机初始化来的
            self.b = tf.get_variable("b", shape=[self.label_size])       # [label_size]

    def inference(self):
        """embedding-->average-->linear classifier"""
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.title)  # [None,title_length,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)  # [None,title_length,embed_size,1]

        # loop each filter size
        # for each filter, do: convolution-pooling layer, feature shape is 4-d. Feature is a new variable
        # a.create filters, b.conv, c.apply nolinearity(tf.nn.relu), d.max-pooling(tf.nn.max_pool)
        pooled_outputs = []  # 池化输出结果
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.name_scope("convolution-pooling-%s" % filter_size):
                # a.create filter
                filter_shape = [filter_size, self.embed_size, 1, self.num_filters]
                # 初始化filter的权重
                filter = tf.get_variable("filter-%s" % filter_size, filter_shape, initializer=self.initializer)
                # b.conv operation
                # *num_filters--->[1,sequence_length-filter_size+1,1,num_filters]
                # *batch_size--->[batch_size,sequence_length-filter_size+1,1,num_filters]
                # conv2d ===> computes a 2-D convolution tensor given 4-D `input` and `filter` tensors.
                # conv2d函数的参数：input: [batch, in_height, in_width, in_channels]，
                #                 filter: [filter_height, filter_width, in_channels, out_channels]
                # output: 4-D [1,sequence_length-filter_size+1,1,1]，得到的是w.x的部分的值
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # c. apply nolinearity
                # h是最终卷积层的输出，即每个feature map，shape=[batch_size,sequence_length-filter_size+1,1,num_filters]
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")
                # d.max-pooling.
                # 输出: 4-D `Tensor` [batch, 1, 1, num_filters]
                # ksize: 想定义多大的范围来进行max-pooling
                # pooled存储的是当前filter_size下每个sentence最重要的num_filters个features
                pooled = tf.nn.max_pool(h, ksize=[1, self.title_length-filter_size+1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")
                pooled_outputs.append(pooled)
        # combine all pooled science_auto_features, and flatten the feature.output' shape is a [batch, 300]
        self.h_pool = tf.concat(pooled_outputs, 3)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])  # [batch,num_filters_total]

        # add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]

        # logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(self.h_drop, self.w) + self.b  # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits

    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)  # sigmoid_cross_entropy_with_logits.#losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y,logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op
