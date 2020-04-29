#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author   : Joshua_Zou
@Contact  : joshua_zou@163.com
@Time     : 2018/3/19 15:34
@Software : PyCharm
@File     : cnn_mlp.py
@Desc     :手写数字识别（多层卷积网络）
"""

import tensorflow as tf
from image_tasks.mnist_digit_recognition.data_processing import *

#读取训练数据
train_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_img"
train_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_lab\train_label.txt"
images = img2mat(train_img_savefile, filenum=60000)
labels = lab2mat(train_lab_savefile, rownum=60000, num_classes=10, one_hot=True)
train = dataset(images,labels)
#读取测试数据
test_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_img"
test_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_lab\test_label.txt"
t_images = img2mat(test_img_savefile, filenum=10000)
t_labels = lab2mat(test_lab_savefile, rownum=10000, num_classes=10, one_hot=True)


#Parameters参数
learning_rate = 1e-4
training_iters = 100000
batch_size = 100
display_step = 10
#Network Parameters
n_input = 784 #MNIST data input(image shape = [28,28])
n_classes = 10 #MNIST total classes (0-9digits)
keep_prob = 0.75 # probability to keep units

#tf Graph input
x = tf.placeholder(tf.float32,[None,n_input])
y_ = tf.placeholder(tf.float32,[None,n_classes])
keep_prob = tf.placeholder(tf.float32)#drop(keep probability)


#初始化权重

def weight_variable(shape):
    initial = tf.truncated_normal(shape,stddev = 0.1)
    return tf.Variable(initial)
def bias_variable(shape):
    # 用稍大于0的值来初始化偏置能够避免节点输出恒为0的问题
    initial = tf.constant(0.1,shape = shape)
    return tf.Variable(initial)

#卷积和池化
def conv2d(x,W):
    return tf.nn.conv2d(x,W,strides = [1,1,1,1],padding = "SAME")

def max_pool_2x2(x):
    return tf.nn.max_pool(x,ksize = [1,2,2,1],strides = [1,2,2,1],padding = "SAME")

#第一层卷积层
with tf.name_scope("conv_layer1"):
    with tf.name_scope("w1"):
        W_conv1 = weight_variable([5,5,1,32])
        tf.summary.histogram("conv_layer1/w1",W_conv1)
    with tf.name_scope("b1"):
        b_conv1 = bias_variable([32])
        tf.summary.histogram("conv_layer1/b1", b_conv1)
    x_image = tf.reshape(x,[-1,28,28,1])
    with tf.name_scope("h_conv1"):
        h_conv1 = tf.nn.relu(conv2d(x_image,W_conv1) + b_conv1)
        tf.summary.histogram("conv_layer1/h_conv1", h_conv1)
    with tf.name_scope("h_pool1"):
        h_pool1 = max_pool_2x2(h_conv1)
    tf.summary.histogram("conv_layer1/h_pool11", h_pool1)

#第二层卷积层
with tf.name_scope("conv_layer2"):
    with tf.name_scope("w2"):
        W_conv2 = weight_variable([5,5,32,64])
        tf.summary.histogram("conv_layer2/w2", W_conv2)
    with tf.name_scope("b2"):
        b_conv2 = bias_variable([64])
        tf.summary.histogram("conv_layer2/b2", b_conv2)
    with tf.name_scope("h_conv2"):
        h_conv2 = tf.nn.relu(conv2d(h_pool1,W_conv2) + b_conv2)
        tf.summary.histogram("conv_layer2/h_conv2", h_conv2)
    with tf.name_scope("h_pool2"):
        h_pool2 = max_pool_2x2(h_conv2)
        tf.summary.histogram("conv_layer2/h_pool2",h_pool2)

#全连接层
with tf.name_scope("fc_layer1"):
    with tf.name_scope("w_fc1"):
        W_fc1 = weight_variable([7*7*64,1024])
        tf.summary.histogram("fc_layer1/w_fc1",W_fc1)
    with tf.name_scope("b_fc1"):
        b_fc1 = bias_variable([1024])
        tf.summary.histogram("fc_layer1/b_fc1",b_fc1)
    h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])
    with tf.name_scope("h_fc1"):
        h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat,W_fc1) + b_fc1)
        tf.summary.histogram("hc_layer/h_fc1",h_fc1)
    #减少过拟合，在输出层之前加上dropout,在训练时启用dropout，测试集的时候关闭
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob=keep_prob)

#输出层
with tf.name_scope("output_layer"):
    with tf.name_scope("w_fc2"):
        W_fc2 = weight_variable([1024,10])
        tf.summary.histogram("output_layer/w_fc2",W_fc2)
    with tf.name_scope("b_fc2"):
        b_fc2 = bias_variable([10])
        tf.summary.histogram("output_layer/b_fc2",b_fc2)
    with tf.name_scope("y_predict"):
        y_pre = tf.nn.softmax(tf.matmul(h_fc1_drop,W_fc2) + b_fc2)

#训练和评估模型
with tf.name_scope("Loss"):
    cross_entropy = - tf.reduce_sum(y_*tf.log(y_pre))
    tf.summary.scalar("Loss",cross_entropy)
with tf.name_scope("train"):
    train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
with tf.name_scope("Accuracy"):
    correct_prediction =tf.equal(tf.argmax(y_pre,1),tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction,tf.float32))
    tf.summary.scalar("Accuracy",accuracy)

#模型可视化
with tf.Session() as sess:
    merged = tf.summary.merge_all()  # 将图形和训练数据合并到一起
    # 创建writer
    log_path = r"D:\Python\Project\TensorFlow\minst\minst_cnn\train"
    train_writer = tf.summary.FileWriter(log_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    step = 1
    while step * batch_size < training_iters:
        batch_xs, batch_ys = train.next_batch(batch_size)
        #run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5},run_metadata = run_metadata)
        if step % display_step == 0:
            # acc = sess.run(accuracy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            # loss = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            # summary = sess.run(merged, feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            acc, loss, summary= sess.run([accuracy, cross_entropy, merged], feed_dict={x: batch_xs, y_: batch_ys, keep_prob: 0.5})
            train_writer.add_summary(summary, step)
            print("Iter " + str(step * batch_size) + ", Minibatch Loss= " + "{:.6f}".format(loss) + ", Training Accuracy= " + "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")
    print("Testing Accuracy:", sess.run(accuracy, feed_dict={x: t_images[:256], y_: t_labels[:256], keep_prob: 1.}))

