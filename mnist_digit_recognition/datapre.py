#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-17 下午4:52
@File    : datapre.py
@Desc    : 

"""

import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


#initialize
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
print(mnist.train)
print(mnist.train.images.shape, mnist.train.labels.shape)
print(mnist.train.images)
# x = tf.placeholder(tf.float32, [None, 784])
# W = tf.Variable(tf.zeros([784, 10]))
# b = tf.Variable(tf.zeros([10]))
# y = tf.nn.softmax(tf.matmul(x, W) + b)
# y_ = tf.placeholder("float", [None, 10])
# cross_entropy = -tf.reduce_sum(y_*tf.log(y))
# train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
# #train
# init = tf.initialize_all_variables()
# sess = tf.Session()
# sess.run(init)
# for i in range(1000):
#     batch_xs, batch_ys = mnist.train.next_batch(100)
#     sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
# #predict
# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
