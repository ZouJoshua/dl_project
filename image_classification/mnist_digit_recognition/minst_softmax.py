#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author   : Joshua_Zou
@Contact  : joshua_zou@163.com
@Time     : 2018/3/15 12:16
@Software : PyCharm
@File     : minst_softmax.py
@Desc     :用softmax激活函数实现手写数字识别
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from minst.data_processing import *

x = tf.placeholder(tf.float32, [None, 784],name = "x-input")
y_ = tf.placeholder(tf.float32,[None,10],name = "y-input")
W = tf.Variable(tf.zeros([784,10]),name = "W")
b = tf.Variable(tf.zeros([10]),name = "b")
y = tf.nn.softmax(tf.matmul(x,W) + b)

#损失函数，交叉熵函数
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
#用梯度下降算法以0.01的学习率最小化交叉熵
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)


#初始化变量
init = tf.global_variables_initializer()
#启动模型
sess = tf.Session()
sess.run(init)
#模型循环训练1000次,选用100个批处理数据点用随机梯度下降训练
train_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_img"
train_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_lab\train_label.txt"
images = img2mat(train_img_savefile,filenum=60000)
labels = lab2mat(train_lab_savefile,rownum = 60000,num_classes=10,one_hot=True)
train = dataset(images,labels)

tf.summary.scalar("Loss",cross_entropy)
merged = tf.summary.merge_all()

#创建writer
log_path = r"D:\Python\Project\TensorFlow\minst\minst_softmax\train"
train_writer = tf.summary.FileWriter(log_path, sess.graph)
#训练及可视化
for i in range(1000):
    batch_xs, batch_ys = train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    summary = sess.run(merged,feed_dict = {x:batch_xs,y_:batch_ys})
    train_writer.add_summary(summary,i)
    if i % 50 ==0:
        train_writer.flush()
train_writer.close()

#计算并打印出正确率
test_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_img"
test_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_lab\test_label.txt"
t_images = img2mat(test_img_savefile,filenum = 10000)
t_labels = lab2mat(test_lab_savefile,rownum = 10000,num_classes=10,one_hot=True)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        #检测我们的预测是否真实标签匹配,分别将预测和真实的标签中取出最大值的索引，相同则返回1(true),不同则返回0(false)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        # 求均值即为准确率
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print(sess.run(accuracy, feed_dict={x: t_images, y_: t_labels}))
tf.summary.scalar('accuracy', accuracy)

