#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/16/19 3:01 PM
@File    : tf2_3.py
@Desc    : 优化参数（学习率）

"""


import tensorflow as tf


LEARNING_RATE_BASE = 0.1  # 初始学习率
LEARNING_RATE_DECAY = 0.99  # 学习率衰减率
LEARNING_RATE_STEP = 1  # 喂入多少轮batch_size后，更新一次学习率，一般为：总样本/batch_size

# 运行几轮batch_size的计数器， 初值为0，设为不被训练
global_step = tf.Variable(0, trainable=False)

# 定义指数下降学习率
learning_rate = tf.train.exponential_decay(LEARNING_RATE_BASE,
                                           global_step,
                                           LEARNING_RATE_STEP,
                                           LEARNING_RATE_DECAY, staircase=True)
# 定义待优化参数，初值为5
w = tf.Variable(5, dtype=tf.float32)

# 定义损失函数
loss = tf.square(w + 1)

# 定义反向传播过程
train_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

# 训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        learning_rate_val = sess.run(learning_rate)
        global_step_val = sess.run(global_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: "
              "global_step is %f, "
              "w is %f, "
              "learning_rate is %f, "
              "loss is %f" % (i, global_step_val, w_val, learning_rate_val, loss_val))