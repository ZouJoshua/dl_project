#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/18/19 3:16 PM
@File    : tf2_5.py
@Desc    : 优化参数（正则化）

"""


"""
正则化在损失函数中引入模型复杂度指标，
利用给w加权值，弱化了训练数据的噪声
loss = loss(y - y_) + REGRULARIZER * loss(w)
"""

"""
数据X[x0, x1]为正态分布，标注Y_当x0^2+x1^2 <2时，y_=1(红)，其余y_=0(蓝)
"""


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt



BATCH_SIZE = 30
seed = 2

# 1.基于seed产生随机数，生成模拟数据集

rdm = np.random.RandomState(seed)
# 随机数返回300行2列的矩阵，表示300组坐标点（x0，x1）作为输入数据集
X = rdm.randn(300, 2)
# print(X)
# 从300行2列的矩阵中取出一行，判断如果两个数据的平方和小于2，给Y赋值1，其余赋值0
# 作为输入数据集的标签
Y_ = [int(x1*x1+x2*x2 < 2) for (x1,x2) in X]
# 遍历Y中的每个元素，1赋值’red‘，其余赋值’blue‘
Y_c = [['red' if i else 'blue'] for i in Y_]
# print(Y_c)
# 整理数据集，Xn行2列，Yn行1列
X = np.vstack(X).reshape(-1, 2)
Y_ = np.vstack(Y_).reshape(-1, 1)

# 2.画出数据集
plt.scatter(X[:,0], X[:, 1], c=np.squeeze(Y_c))
plt.show()


# 3.定义神经网络的输入、参数和输出，定义前向传播过程
def get_weight(shape, regularizer):
    w = tf.Variable(tf.random_normal(shape), dtype=tf.float32)
    tf.add_to_collection("losses", tf.contrib.layers.l2_regularizer(regularizer)(w))
    return w

def get_bias(shape):
    b = tf.Variable(tf.constant(0.01, shape=shape))
    return b

x = tf.placeholder(tf.float32, shape=(None, 2))
y_ = tf.placeholder(tf.float32, shape=(None, 1))

w1 = get_weight([2, 11], 0.01)
b1 = get_bias([11])

y1 = tf.nn.relu(tf.matmul(x, w1)+b1)

w2 = get_weight([11, 1], 0.01)
b2 = get_bias([1])
y = tf.matmul(y1, w2)+b2  # 输出层不过激活函数

# 4.定义损失函数
loss_mse = tf.reduce_mean(tf.square(y-y_))
loss_total = loss_mse + tf.add_n(tf.get_collection("losses"))

# 5.定义反向传播方法，不含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_mse)
with tf.Session() as sess:
    print("未添加正则化的训练过程...")
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (BATCH_SIZE*i) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 ==0:
            loss_mse_val = sess.run(loss_mse, feed_dict={x: X, y_: Y_})
            print("After %d steps, loss is: %f" % (i, loss_mse_val))
    # xx在-3到3之间以步长0.01， yy在-3到3之间以步长0.01，生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
    # 将xx、yy拉直， 并合并成一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网格坐标点喂入神经网络，probs为输出
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)
    print("w1:\n", sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))
    print("b2:\n", sess.run(b2))
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx,yy,probs, levels=[.5])
plt.show()


# 6.定义反向传播方法，含正则化
train_step = tf.train.AdamOptimizer(0.0001).minimize(loss_total)
with tf.Session() as sess:
    print("添加正则化的训练过程...")
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    STEPS = 40000
    for i in range(STEPS):
        start = (BATCH_SIZE*i) % 300
        end = start + BATCH_SIZE
        sess.run(train_step, feed_dict={x: X[start:end], y_: Y_[start:end]})
        if i % 2000 == 0:
            loss_total_val = sess.run(loss_total, feed_dict={x: X, y_: Y_})
            print("After %d steps, loss is: %f" % (i, loss_total_val))
    # xx在-3到3之间以步长0.01， yy在-3到3之间以步长0.01，生成二维网格坐标点
    xx, yy = np.mgrid[-3:3:0.01, -3:3:0.01]
    # 将xx、yy拉直， 并合并成一个2列的矩阵，得到一个网格坐标点的集合
    grid = np.c_[xx.ravel(), yy.ravel()]
    # 将网格坐标点喂入神经网络，probs为输出
    probs = sess.run(y, feed_dict={x: grid})
    probs = probs.reshape(xx.shape)
    print("w1:\n", sess.run(w1))
    print("b1:\n", sess.run(b1))
    print("w2:\n", sess.run(w2))
    print("b2:\n", sess.run(b2))
plt.scatter(X[:, 0], X[:, 1], c=np.squeeze(Y_c))
plt.contour(xx,yy,probs, levels=[.5])
plt.show()