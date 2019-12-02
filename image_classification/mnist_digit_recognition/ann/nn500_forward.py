#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/18/19 10:54 PM
@File    : nn_forward.py
@Desc    : 单隐藏层500结点前向传播

"""


import tensorflow as tf


"""
网络结构：
输入层 -> 隐藏层1（500神经元） -> 输出层
Layer1:
input[None,784] -> relu[784, 500] -> relu[784, 500]
Layer2:
[500,10] -> softmax[None,10]

"""

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 500


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1 = tf.nn.relu(tf.matmul(x, w1) + b1)

    w2 = get_weight([LAYER1_NODE, OUTPUT_NODE], regularizer)
    b2 = get_bias([OUTPUT_NODE])

    y = tf.matmul(y1, w2) + b2

    return y
