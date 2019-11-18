#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/16/19 12:25 PM
@File    : tf1_4.py
@Desc    : 反向传播(训练模型参数，在所有参数上梯度下降，使模型在训练数据上损失最小)

"""


import tensorflow as tf
import numpy as np


BATCH_SIZE = 8
seed = 23455

# 基于seed生成随机数
rng = np.random.RandomState(seed)
# 随机数返回32行2列的矩阵（表示32组体积和重量，作为输入数据集）
X = rng.rand(32, 2)
# 从X中取出一行，判断如果和小于1，给Y赋值1，否则赋值0，作为输入数据集的标签Y
Y = [[int(x0 + x1 < 1)] for (x0, x1) in X]


# 定义神经网络的输入，参数和输出，定义前向传播过程
x = tf.placeholder(tf.float32, shape=[None, 2])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)

# 定义损失函数及反向传播
loss = tf.reduce_mean(tf.square(y-y_))
train_step = tf.train.GradientDescentOptimizer(0.001).minimize(loss)
# train_step = tf.train.MomentumOptimizer(0.001, 0.9).minimize(loss)
# train_step = tf.train.AdamOptimizer(0.001).minimize(loss)


# 生成会话，训练STEPS轮

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输入未经训练的参数取值
    print("初始化:")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))

    # 训练模型
    STEPS = 30000
    for i in range(STEPS):
        start = (i*BATCH_SIZE) % 32
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 500 == 0:
            total_loss = sess.run(loss, feed_dict={x: X, y_: Y})
            print("After {} training step(s), loss on all data is {}".format(i, total_loss))
    # 训练后参数
    print("训练后:")
    print("w1:\n", sess.run(w1))
    print("w2:\n", sess.run(w2))
