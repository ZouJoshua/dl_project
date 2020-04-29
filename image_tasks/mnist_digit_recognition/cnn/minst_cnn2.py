#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author   : Joshua_Zou
@Contact  : joshua_zou@163.com
@Time     : 2018/2/28 15:04
@Software : PyCharm
@File     : minst_cnn.py
@Desc     :
"""

"""
流程如下：
Layer1：
input【28x28x1】->conv【5x5，(1:32)】->relu【28x28x32】->max_pool【2x2，14x14x32】->dropout【0.75】
Layer2：
【14x14x32】->conv【5x5，(32:64)】->relu【14x14x64】->max_pool【2x2，7x7x64】->dropout【0.75】
Layer3：
【7x7x64】->FC【1024】->relu->dropout->y = wx+b【10】
"""


import tensorflow as tf
from image_tasks.mnist_digit_recognition.data_processing import *
from numpy import outer

#读取训练数据
train_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_img"
train_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_lab\train_label.txt"
images = img2mat(train_img_savefile,filenum=60000)
labels = lab2mat(train_lab_savefile,rownum = 60000,num_classes=10,one_hot=True)
train = dataset(images,labels)
#读取测试数据
test_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_img"
test_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_lab\test_label.txt"
t_images = img2mat(test_img_savefile,filenum = 10000)
t_labels = lab2mat(test_lab_savefile,rownum = 10000,num_classes=10,one_hot=True)

#Parameters参数
learning_rate = 0.001
training_iters = 100000
batch_size = 128
display_step = 10
#Network Parameters
n_input = 784 #MNIST data input(image shape = [28,28])
n_classes = 10 #MNIST total classes (0-9digits)
dropout = 0.75 # probability to keep units

#tf Graph input
x = tf.placeholder(tf.float32,[None,n_input])
y = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)#drop(keep probability)

#Create model
def conv2d(image,w,b):
    return tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(image,w,strides=[1,1,1,1],padding='SAME'),b))
def max_pooling(image,k):
    return tf.nn.max_pool(image, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

weights = {
    'wc1':tf.Variable(tf.random_normal([5,5,1,32])),
    'wc2':tf.Variable(tf.random_normal([5,5,32,64])),
    'wd1':tf.Variable(tf.random_normal([7*7*64,1024])),
    'out':tf.Variable(tf.random_normal([1024,n_classes]))
}
biases = {
    'bc1':tf.Variable(tf.random_normal([32])),
    'bc2':tf.Variable(tf.random_normal([64])),
    'bd1':tf.Variable(tf.random_normal([1024])),
    'out':tf.Variable(tf.random_normal([n_classes]))
}
def conv_net(_X,_weights,_biases,_dropout):
    #Layer 1
    _X = tf.reshape(_X,[-1,28,28,1])
    conv1 = conv2d(_X,_weights['wc1'],_biases['bc1'])
    conv1 = max_pooling(conv1, k = 2)
    conv1 = tf.nn.dropout(conv1, keep_prob=_dropout)
    #Layer 2
    conv2 = conv2d(conv1,_weights['wc2'],_biases['bc2'])
    conv2 = max_pooling(conv2, k=2)
    conv2 = tf.nn.dropout(conv2,keep_prob=_dropout)
    #Fully Connected
    dense1 = tf.reshape(conv2,[-1,_weights['wd1'].get_shape().as_list()[0]])
    dense1 = tf.nn.relu(tf.add(tf.matmul(dense1,_weights['wd1']),_biases['bd1']))
    dense1 = tf.nn.dropout(dense1,_dropout)
    out = tf.add(tf.matmul(dense1,_weights['out']),_biases['out'])
    print(out)
    return out
#Construct model
pred = conv_net(x, weights, biases, keep_prob)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

#Evaluate model
correct_pred = tf.equal(tf.argmax(pred,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(correct_pred,tf.float32))
init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    step = 1
    while step * batch_size<training_iters:
        batch_xs,batch_ys = train.next_batch(batch_size)
        sess.run(optimizer,feed_dict = {x:batch_xs,y:batch_ys,keep_prob:dropout})
        if step %display_step==0:
            acc = sess.run(accuracy,feed_dict={x:batch_xs,y:batch_ys,keep_prob:1.})
            loss = sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, keep_prob: 1.})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))

        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy:",sess.run(accuracy,feed_dict={x: t_images[:256], y: t_labels[:256], keep_prob: 1.}))