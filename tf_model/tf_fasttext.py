#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-14 下午5:55
@File    : tf_fasttext.py
@Desc    : 
"""

import tensorflow as tf

class fastText(object):

    def __init__(self,
                 label_size,
                 learning_rate,
                 batch_size,
                 learning_decay_rate,
                 learning_decay_steps,
                 num_sampled,
                 text_len,
                 vocab_size,
                 embedding_dims,
                 keep_prob,
                 is_training,
                 max_label_per_example=5):

        """初始化超参数"""
        # 1.set hyper-paramter
        self.label_size = label_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.text_len = text_len
        self.vocab_size = vocab_size
        self.embed_size = embedding_dims
        self.is_training = is_training
        self.learning_rate = learning_rate
        self.dropout_keep_prob = keep_prob
        self.max_label_per_example = max_label_per_example
        self.initializer = tf.random_normal_initializer(stddev=0.1)

        # 2.add placeholder (X,label)
        self.text = tf.placeholder(tf.int32, [None, self.text_len], name="text")
        self.labels = tf.placeholder(tf.int64, [None, self.max_label_per_example], name="labels")
        self.labels_l1999 = tf.placeholder(tf.float32, [None, self.label_size])  # int64

        # 3.set some variables
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.decay_steps, self.decay_rate = learning_decay_steps, learning_decay_rate
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")

        # 4.init weights
        self.init_weights()

        # 5.main graph: inference
        self.logits = self.inference()  # [None, self.label_size]

        # 6.calculate loss
        self.loss_val = self.loss()

        # 7.start training by update parameters using according loss
        self.train_op = self.train()

        # 8.calcuate accuracy
        self.accuracy = self.accuracy()

    def init_weights(self):
        """define all weights here"""
        # embedding matrix
        with tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embed_size], initializer=self.initializer)
        self.W = tf.get_variable("W", [self.embed_size, self.label_size], initializer=self.initializer)
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        """计算图: 1.embedding-->2.average-->3.linear classifier"""
        # 1.get emebedding of words in the sentence
        embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.text)  # [None,self.sentence_len,self.embed_size]

        with tf.name_scope("dropout"):
            dropout_output = tf.nn.dropout(embedding_inputs, self.dropout_keep_prob)

        # 2.average vectors, to get representation of the sentence
        with tf.name_scope("average"):
            # self.inputs_embeddings = tf.reduce_mean(dropout_output, axis=1)
            self.inputs_embeddings = tf.reduce_mean(embedding_inputs, axis=1)  # [None,self.embed_size]

        # 3.linear classifier layer
        logits = tf.matmul(self.inputs_embeddings, self.W) + self.b  # [None, self.label_size]==tf.matmul([None,self.embed_size],[self.embed_size,self.label_size])
        return logits


    def loss(self, l2_lambda=0.0001):
        """计算NCE交叉熵损失"""
        if self.is_training:
            labels = tf.reshape(self.labels, [-1])
            labels = tf.expand_dims(labels, 1)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.W),
                               biases=self.b,
                               labels=labels,
                               inputs=self.inputs_embeddings,
                               num_sampled=self.num_sampled,
                               num_classes=self.label_size,
                               partition_strategy="div")
            )
        else:
            labels_one_hot = tf.one_hot(self.labels, self.label_size)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
            print("loss0:", losses)
            loss = tf.reduce_sum(losses, axis=1)
            print("loss1:", loss)

        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def accuracy(self):
        with tf.name_scope('accuracy'):
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int64), self.labels)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        return accuracy

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam")
        return train_op
