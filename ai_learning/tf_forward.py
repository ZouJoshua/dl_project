#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/18/19 5:59 PM
@File    : tf_forward.py
@Desc    : 前向传播过程

"""


import tensorflow as tf



def forward(x, regularizer):
    w = None
    b = None
    y = tf.matmul(x, w) + b
    return y

def get_weight(shape, regularizer):
    w = tf.Variable(shape, dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(shape, dtype=tf.float32)
    return b