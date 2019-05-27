#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:21
@File    : fasttext_model.py
@Desc    : 

"""

import tensorflow as tf


class FastText(object):
    
    def __init__(self,
                 label_size,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 batch_size,
                 num_sampled,
                 dropout_keep_prob,
                 sentence_len,
                 vocab_size,
                 embed_size,
                 is_training):
        self.label_size = label_size  # num of classes
        self.learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.learning_decay_rate = learning_decay_rate
        self.learning_decay_steps = learning_decay_steps
        self.vocab_size = vocab_size
        self.dropout_keep_prob = dropout_keep_prob
        self.embedding_dims = embed_size
        self.batch_size = batch_size
        self.num_sampled = num_sampled
        self.sentence_len = sentence_len
        self.is_training = is_training

        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")   # X
        self.label = tf.placeholder(tf.int32, [None], name="label")  # Y

        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        self.global_step = tf.Variable(0, trainable=False, name="global_step")

        self.init_weights()
        self.logits = self.inference()
        self.loss_val = self.l2_loss()
        self.train_op = self.train()
        self.accuracy = self.acc()

    def init_weights(self):
        with tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dims])
        self.w = tf.get_variable("w", [self.embedding_dims, self.label_size])
        self.b = tf.get_variable("b", [self.label_size])

    def inference(self):
        """计算图：embedding -> average -> linear classifier"""
        embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.sentence)

        with tf.name_scope("dropout"):
            dropout_output = tf.nn.dropout(embedding_inputs, self.dropout_keep_prob)

        # 对词向量进行平均
        with tf.name_scope("average"):
            # self.inputs_embeddings = tf.reduce_mean(dropout_output, axis=1)
            self.embeddings_output = tf.reduce_mean(embedding_inputs, axis=1)
        # 输出层
        logits = tf.matmul(self.embeddings_output, self.w) + self.b
        return logits

    def nce_loss(self, l2_lambda=0.001):
        """计算NCE交叉熵损失"""
        if self.is_training:
            labels = tf.reshape(self.label, [-1])
            labels = tf.expand_dims(labels, 1)
            loss = tf.reduce_mean(
                tf.nn.nce_loss(weights=tf.transpose(self.w),
                               biases=self.b,
                               labels=labels,
                               inputs=self.embeddings_output,
                               num_sampled=self.num_sampled,
                               num_classes=self.label_size,
                               partition_strategy="div")
            )
        else:
            labels_one_hot = tf.one_hot(self.label, self.label_size)
            losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
            print("loss0:", losses)
            loss = tf.reduce_sum(losses, axis=1)
            print("loss1:", loss)

        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def l2_loss(self, l2_lambda=0.001):
        """
        根据每次训练的预测结果和标准结果比较，计算误差
                loss = loss + l2_lambda*1/2*||variables||2
        :param l2_lambda: 超参数，l2正则，保证l2_loss和train_loss在同一量级
        :return: 每次训练的损失值loss
        """
        self.y_true = tf.one_hot(self.label, self.label_size)
        # self.y_true = labels_one_hot.eval()
        losses = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.y_true, logits=self.logits)
        # losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels_one_hot, logits=self.logits)
        # losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
        loss = tf.reduce_mean(losses)
        l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
        loss = loss + l2_losses
        return loss

    def acc(self):
        with tf.name_scope('accuracy'):
            self.predictions = tf.argmax(self.logits, axis=1, name="predictions")
            self.y_pred = tf.one_hot(self.predictions, self.label_size)
            correct_predictions = tf.equal(tf.cast(self.predictions, tf.int32), self.label)
            accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32), name="accuracy")
        return accuracy

    def train(self):
        learining_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.learning_decay_steps,
                                                    self.learning_decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step,
                                                   learning_rate=learining_rate, optimizer="Adam")
        return train_op
