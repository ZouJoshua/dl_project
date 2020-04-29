#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/4/19 5:35 PM
@File    : func_utils.py
@Desc    : 自定义函数

"""


import tensorflow as tf


def get_weight(shape, regularizer):
    w = tf.Variable(tf.truncated_normal(shape, stddev=0.1))
    if regularizer != None:
        tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w


def get_bias(shape):
    b = tf.Variable(tf.zeros(shape))
    return b


def fcn_layer(input,  # 输入数据
              input_dim,  # 输入神经元数量
              output_dim,  # 输出神经元数量
              activation=None):  #激活函数

    w = tf.Variable(tf.truncated_normal([input_dim, output_dim], stddev=0.1))
    b = tf.Variable(tf.zeros([output_dim]))
    h = tf.matmul(input, w) + b

    if activation is None:
        output = h
    else:
        output = activation(h)

    return output