#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/18/19 6:10 PM
@File    : tf_backword.py
@Desc    : 反向传播过程

"""

from ai_learning.tf_forward import forward
import tensorflow as tf


REGULARIZER = 0.01
LEARNING_RATE_BASE = 0.01
LEARNING_RATE_DECAY = 0.99
MOVING_AVERAGE_DECAY = 0.99

N = 10000  # 样本总数
BATCH_SIZE = 100  # 批样本数
STEPS = 50  # 迭代轮数



def bachword():
    x = tf.placeholder(tf.float32, shape=())
    y_ = tf.placeholder(tf.float32, shape=())
    y = forward(x, REGULARIZER)
    global_step = tf.Variable(0, trainable=False)
    mse_loss = tf.reduce_mean(tf.square(y-y_))

    # ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    # ce_loss = tf.reduce_mean(ce)

    # 加入正则化损失
    loss = mse_loss + tf.add_n(tf.get_collection("losses"))
    # loss = ce_loss + tf.add_n(tf.get_collection("losses"))


    # 指数衰减学习率
    learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE, global_step, N/BATCH_SIZE, LEARNING_RATE_DECAY, staircase=True)

    train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)


    # 滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables)
    with tf.control_dependencies([train_step, ema_op]):
        train_op = tf.no_op(name='train')

    with tf.Session() as sess:
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        for i in range(STEPS):
            start = (BATCH_SIZE*i) % N
            end = start + BATCH_SIZE
            # sess.run(train_step, feed_dict={x: x[start:end],y_: y[start:end]})
            sess.run(train_op, feed_dict={x: x[start:end],y_: y[start:end]})
            if i % 300 == 0:
                print("")
