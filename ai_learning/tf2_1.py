#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/16/19 1:56 PM
@File    : tf2_1.py
@Desc    : 优化参数（损失函数）

"""

import tensorflow as tf
import numpy as np


BATCH_SIZE = 8
seed = 23455

# 基于seed生成随机数
rng = np.random.RandomState(seed)
# 随机数返回32行2列的矩阵（表示32组体积和重量，作为输入数据集）
X = rng.rand(32, 2)
# 模拟数据集y_=x1+x2, 引入噪声：-0.05~+0.05
Y = [[x1+x2+(rng.rand()/10.0-0.05)] for (x1, x2) in X]


# 定义神经网络的输入，参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = tf.Variable(tf.random_normal([2, 1], stddev=1, seed=1))

y = tf.matmul(x, w1)

# 定义损失函数MSE,反向传播算法为的梯度下降
loss = tf.reduce_mean(tf.square(y_-y))
# 自定义损失函数
# 交叉熵损失函数

train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)

# 生成会话，训练STEPS轮

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输入未经训练的参数取值
    print("初始化:")
    print("w1:\n", sess.run(w1))

    # 训练模型
    STEPS = 30000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss, w_1 = sess.run([loss, w1], feed_dict={x: X, y_: Y})
            print("After {} training step(s), loss on all data is {}, w1 is :\n{}".format(i, total_loss, w_1))
    # 训练后参数
    print("训练后:")
    print("w1:\n", sess.run(w1))
