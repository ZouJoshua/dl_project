#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/18/19 10:54 PM
@File    : dnn_forward.py
@Desc    : 双隐藏层前向传播

"""


import tensorflow as tf


"""
网络结构：
输入层 -> 隐藏层1（600神经元） -> 隐藏层2(480神经元) -> 输出层
Layer1:
input[None,784] -> relu[784, 600] -> dropout[600, 480]
Layer2:
relu[600,480] -> dropout[480, 10] -> softmax[None,10]

"""

INPUT_NODE = 784
OUTPUT_NODE = 10
LAYER1_NODE = 600
LAYER2_NODE = 480


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def forward(x, regularizer, dropout):
    w1 = get_weight([INPUT_NODE, LAYER1_NODE], regularizer)
    b1 = get_bias([LAYER1_NODE])
    y1_ = tf.nn.relu(tf.matmul(x, w1) + b1)
    y1 = tf.nn.dropout(y1_, dropout)

    w2 = get_weight([LAYER1_NODE, LAYER2_NODE], regularizer)
    b2 = get_bias([LAYER2_NODE])
    y2_ = tf.nn.relu(tf.matmul(y1, w2) + b2)
    y2 = tf.nn.dropout(y2_, dropout)

    w = get_weight([LAYER2_NODE, OUTPUT_NODE], regularizer)
    b = get_bias(OUTPUT_NODE)
    y = tf.matmul(y2, w) + b
    return y
