#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-14 下午5:55
@File    : tf_fasttext.py
@Desc    : 
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow.contrib import slim
from eval.evaluate import accuracy
from loss.loss import cross_entropy_loss


class FastText(object):
    def __init__(self,
                 num_classes,
                 seq_length,
                 vocab_size,
                 embedding_dim,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 epoch,
                 dropout_keep_prob):
        self.num_classes = num_classes
        self.seq_length = seq_length
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.learning_rate = learning_rate
        self.learning_decay_rate = learning_decay_rate
        self.learning_decay_steps = learning_decay_steps
        self.epoch = epoch
        self.dropout_keep_prob = dropout_keep_prob
        self.input_x = tf.placeholder(tf.int32, [None, self.seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, self.num_classes], name='input_y')
        self.model()

    def model(self):
        # 词向量映射
        with tf.name_scope("embedding"):
            self.embedding = tf.get_variable("embedding", [self.vocab_size, self.embedding_dim])
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("dropout"):
            dropout_output = tf.nn.dropout(embedding_inputs, self.dropout_keep_prob)

        # 对词向量进行平均
        with tf.name_scope("average"):
            mean_sentence = tf.reduce_mean(dropout_output, axis=1)

        # 输出层
        with tf.name_scope("score"):
            self.logits = tf.layers.dense(mean_sentence, self.num_classes, name='dense_layer')

        # 损失函数
        self.loss = cross_entropy_loss(logits=self.logits, labels=self.input_y)

        # 优化函数
        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step,
                                                   self.learning_decay_steps, self.learning_decay_rate,
                                                   staircase=True)

        optimizer = tf.train.AdamOptimizer(learning_rate)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        self.optim = slim.learning.create_train_op(total_loss=self.loss, optimizer=optimizer, update_ops=update_ops)

        # 准确率
        self.acc = accuracy(logits=self.logits, labels=self.input_y)

    def fit(self, train_x, train_y, val_x, val_y, batch_size):
        # 创建模型保存路径
        if not os.path.exists('./saves/fasttext'): os.makedirs('./saves/fasttext')
        if not os.path.exists('./train_logs/fasttext'): os.makedirs('./train_logs/fasttext')

        # 开始训练
        train_steps = 0
        best_val_acc = 0
        # summary
        tf.summary.scalar('val_loss', self.loss)
        tf.summary.scalar('val_acc', self.acc)
        merged = tf.summary.merge_all()

        # 初始化变量
        sess = tf.Session()
        writer = tf.summary.FileWriter('./train_logs/fasttext', sess.graph)
        saver = tf.train.Saver(max_to_keep=10)
        sess.run(tf.global_variables_initializer())

        for i in range(self.epoch):
            batch_train = self.batch_iter(train_x, train_y, batch_size)
            for batch_x, batch_y in batch_train:
                train_steps += 1
                feed_dict = {self.input_x: batch_x, self.input_y: batch_y}
                _, train_loss, train_acc = sess.run([self.optim, self.loss, self.acc], feed_dict=feed_dict)

                if train_steps % 1000 == 0:
                    feed_dict = {self.input_x: val_x, self.input_y: val_y}
                    val_loss, val_acc = sess.run([self.loss, self.acc], feed_dict=feed_dict)

                    summary = sess.run(merged, feed_dict=feed_dict)
                    writer.add_summary(summary, global_step=train_steps)

                    if val_acc >= best_val_acc:
                        best_val_acc = val_acc
                        saver.save(sess, "./saves/fasttext/", global_step=train_steps)

                    msg = 'epoch:%d/%d,train_steps:%d,train_loss:%.4f,train_acc:%.4f,val_loss:%.4f,val_acc:%.4f'
                    print(msg % (i, self.epoch, train_steps, train_loss, train_acc, val_loss, val_acc))

    def batch_iter(self, x, y, batch_size=32, shuffle=True):
        """
        生成batch数据
        :param x: 训练集特征变量
        :param y: 训练集标签
        :param batch_size: 每个batch的大小
        :param shuffle: 是否在每个epoch时打乱数据
        :return:
        """
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_len))
            x_shuffle = x[shuffle_indices]
            y_shuffle = y[shuffle_indices]
        else:
            x_shuffle = x
            y_shuffle = y
        for i in range(num_batch):
            start_index = i * batch_size
            end_index = min((i + 1) * batch_size, data_len)
            yield (x_shuffle[start_index:end_index], y_shuffle[start_index:end_index])

    def predict(self, x):
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(tf.global_variables())
        ckpt = tf.train.get_checkpoint_state('./saves/fasttext/')
        saver.restore(sess, ckpt.model_checkpoint_path)

        feed_dict = {self.input_x: x}
        logits = sess.run(self.logits, feed_dict=feed_dict)
        y_pred = np.argmax(logits, 1)
        return y_pred
