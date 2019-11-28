#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author   : Joshua_Zou
@Contact  : joshua_zou@163.com
@Time     : 2018/2/28 15:21
@Software : PyCharm
@File     : minst_ann_mlp.py
@Desc     :用多层感知器学习的人工神经网络
"""
import tensorflow as tf
import data.input_data as input_data
from pyexpat import features
mnist = input_data.read_data_sets("MNIST_data/",one_hot=True)
learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
#NetWork parameters
n_hidden_1 = 256#1st layer num features
n_hidden_2 = 256#2nd layer num features
n_input = 784
n_classses = 10

x = tf.placeholder("float", [None,n_input])
y = tf.placeholder("float",[None,n_classses])

def multilayer_perceptron(_X,_weights,_biases):
    layer1 = tf.nn.sigmoid(tf.add(tf.matmul(_X, _weights['h1']), _biases['b1']))
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1,_weights['h2']),_biases['b2']))
    return tf.matmul(layer2,_weights['out'])+_biases['out']

weights = {
    'h1':tf.Variable(tf.random_normal([n_input,n_hidden_1])),
    'h2':tf.Variable(tf.random_normal([n_hidden_1,n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_hidden_2,n_classses]))
}
biases = {
    'b1':tf.Variable(tf.random_normal([n_hidden_1])),
    'b2':tf.Variable(tf.random_normal([n_hidden_2])),
    'out':tf.Variable(tf.random_normal([n_classses]))
}
pred = multilayer_perceptron(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    #Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train._num_examples/batch_size)
        for i in range(total_batch):
            batch_xs,batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys})/total_batch
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(avg_cost))
    print("Optimization Finished!")
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))