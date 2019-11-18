#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/16/19 11:48 AM
@File    : tf1_2.py
@Desc    : 前向传播过程

"""


import tensorflow as tf


x = tf.constant([[0.7, 0.5]])
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


# 会话计算结果


with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print("the result: \n", sess.run(y))
