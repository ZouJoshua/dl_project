#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/16/19 2:45 PM
@File    : tf2_2.py
@Desc    : 优化参数（学习率）

"""


"""
学习率大了震荡不收敛，学习率小了收敛速度慢
"""

import tensorflow as tf


# 设损失函数为
# loss = (w + 1)^2
# 令w初值为5

# 定义待优化参数w初值为5
w = tf.Variable(tf.constant(5, dtype=tf.float32))

# 定义损失函数
loss = tf.square(w+1)
# 定义反向传播方法

train_step = tf.train.GradientDescentOptimizer(0.2).minimize(loss)

# 训练40轮
with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    for i in range(40):
        sess.run(train_step)
        w_val = sess.run(w)
        loss_val = sess.run(loss)
        print("After %s steps: w is %f, loss is %f" % (i, w_val, loss_val))
