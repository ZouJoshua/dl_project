#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/16/19 11:40 AM
@File    : tf1_1.py
@Desc    : 

"""

import tensorflow as tf



x = tf.constant([[1.0, 2.0]])
w = tf.constant([[3.0], [4.0]])
y = tf.matmul(x, w)


print(y)
with tf.Session() as sess:
    print(sess.run(y))