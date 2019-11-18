#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/16/19 11:48 AM
@File    : tf1_2.py
@Desc    : 前向传播过程

"""


import tensorflow as tf

# 用placeholder定义输入，sess.run中喂入一组数据
# x = tf.placeholder(tf.float32, shape=[1, 2])
# 用placeholder定义输入，sess.run中喂入多组数据
x = tf.placeholder(tf.float32, shape=[None, 2])

w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

# 定义前向传播过程

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)


# 会话计算结果

with tf.Session() as sess:
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    # 输入一组特征数据
    # print("the result: \n", sess.run(y, feed_dict={x: [[0.7, 0.5]]}))
    #输入多组特征数据
    print("the result: \n", sess.run(y, feed_dict={x: [[0.7, 0.5], [0.2, 0.3], [0.3, 0.4]]}))
    print("w1: \n", sess.run(w1))
    print("w2: \n", sess.run(w2))
