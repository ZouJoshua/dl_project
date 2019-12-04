#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/2/19 8:19 PM
@File    : nn_forward.py
@Desc    : 单层正向传播

"""

"""
网络结构：
输入层 -> 输出层
Layer1:
input[None,784] -> softmax[784,10] -> out[None, 10]
"""


import tensorflow as tf


INPUT_NODE = 784
OUTPUT_NODE = 10



def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, OUTPUT_NODE], regularizer)
    b1 = get_bias([OUTPUT_NODE])
    y = tf.nn.softmax(tf.matmul(x, w1) + b1)

    return y