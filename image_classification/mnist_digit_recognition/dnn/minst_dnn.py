#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author   : Joshua_Zou
@Contact  : joshua_zou@163.com
@Time     : 2018/2/28 15:21
@Software : PyCharm
@File     : minst_dnn.py
@Desc     :
"""

import tensorflow as tf
import data.input_data as input_data

mnist = input_data.read_data_sets("MNIST/", one_hot=True)
#Paramters
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10

#Network parameters
n_input = 784
n_classes = 10
dropout = 0.8
#tf Graph input
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)
#Create model
def conv2d(image,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image,w,strides=[1,1,1,1],padding='SAME'),b))
def max_pool(image,k):
    return tf.nn.max_pool(image,ksize=[1,k,k,1],strides=[1,k,k,1],padding='SAME')

def dnn(_X,_weights,_biases,_dropout):
    _X = tf.nn.dropout(_X,_dropout)
    d1 = tf.nn.relu(tf.nn.bias_add(tf.matmul(_X, _weights['wd1']),_biases['bd1']),name='d1')
    d2x = tf.nn.dropout(d1,_dropout)
    d2 = tf.nn.relu(tf.nn.bias_add(tf.matmul(d2x,_weights['wd2']),_biases['bd2']),name='d2')
    dout = tf.nn.dropout(d2,_dropout)
    out = tf.matmul(dout,weights['out'])+_biases['out']
    return out
weights = {
    'wd1':tf.Variable(tf.random_normal([784,600],stddev=0.01)),
    'wd2':tf.Variable(tf.random_normal([600,480],stddev=0.01)),
    'out':tf.Variable(tf.random_normal([480,10]))
}
biases = {
    'bd1':tf.Variable(tf.random_normal([600])),
    'bd2':tf.Variable(tf.random_normal([480])),
    'out':tf.Variable(tf.random_normal([10])),
}
#Construct model
pred = dnn(x, weights, biases, keep_prob)

#Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate = learning_rate).minimize(cost)
#Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))

init = tf.initialize_all_variables()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size < training_iters:
        batch_xs,batch_ys = mnist.train.next_batch(batch_size)
        sess.run(optimizer,feed_dict={x:batch_xs,y:batch_ys,keep_prob:dropout})
        if step % display_step == 0:
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            loss = sess.run(cost,feed_dict = {x:batch_xs,y:batch_ys,keep_prob:1.})
            print("Iter "+str(step*batch_size)+",Minibatch Loss = "+"{:.6f}".format(loss)+", Training Accuracy = "+"{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print("Testing Accuarcy : ",sess.run(accuracy,feed_dict={x:mnist.test.images[:256],y:mnist.test.labels[:256]}))