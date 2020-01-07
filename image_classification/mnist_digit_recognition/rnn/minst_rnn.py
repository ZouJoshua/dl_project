#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author   : Joshua_Zou
@Contact  : joshua_zou@163.com
@Time     : 2018/2/28 15:20
@Software : PyCharm
@File     : minst_rnn.py
@Desc     :循环神经网络实现手写数字识别
"""

import data.input_data as input_data
mnist = input_data.read_data_sets("MNIST_data", one_hot=True)
import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

"""
To classify images using a rnn,we consider every image row as a sequence of pixels
becaues MNIST image shape is 28*28px,we will then handle 28 sequences of 28 steps for every sample
"""
#Parameters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

#Network parameters
n_input = 28
n_steps = 28
n_hidden = 128#hidden layer num
n_classes = 10

#tf Graph input
x = tf.placeholder("float",[None,n_steps,n_input])
#Tensorflow LSTM cell requires 2xn_hidden length(state&cell)
istate = tf.placeholder("float",[None,2*n_hidden])
#output
y = tf.placeholder("float",[None,n_classes])
#random initialize biases and weights
weights = {
    "hidden":tf.Variable(tf.random_normal([n_input,n_hidden])),
    "out":tf.Variable(tf.random_normal([n_hidden,n_classes]))
}
biases = {
    "hidden":tf.Variable(tf.random_normal([n_hidden])),
    "out":tf.Variable(tf.random_normal([n_classes]))
}
#RNN
def RNN(_X,_istate,_weights,_biases):
    _X = tf.transpose(_X,[1,0,2])
    _X = tf.reshape(_X,[-1,n_input])
    #input Layer to hidden Layer
    _X = tf.matmul(_X,_weights['hidden'])+_biases['hidden']
    #LSTM cell
    lstm_cell = rnn.BasicLSTMCell(n_hidden,state_is_tuple=False)
    #28 sequence need to splite 28 time
    _X = tf.split(_X,n_steps,0)
    #start to run rnn
    outputs,states = rnn.static_rnn(lstm_cell,_X,initial_state = _istate)
    return tf.matmul(outputs[-1],weights['out'])+biases['out']

pred = RNN(x,istate,weights,biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels = y))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape((batch_size,n_steps,n_input))
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,
                                      istate:np.zeros((batch_size,2*n_hidden))})
        if step % display_step == 0:
            acc = sess.run(accuracy, feed_dict={x: batch_xs, y: batch_ys,
                                                istate: np.zeros((batch_size, 2 * n_hidden))})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys,
                                             istate: np.zeros((batch_size, 2 * n_hidden))})
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    test_len = 256
    test_data = mnist.test.images[:test_len].reshape((-1,n_steps,n_input))
    test_label = mnist.test.labels[:test_len]
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: test_data, y: test_label,
                                                             istate: np.zeros((test_len, 2 * n_hidden))}))