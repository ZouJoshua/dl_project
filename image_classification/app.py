#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/29/19 10:58 PM
@File    : app.py
@Desc    : 

"""


import numpy as np

# linux服务器没有GUI的情况下使用matplotlib绘图，必须至于pyplot之前
# import matplotlib
# matplotlib.use("Agg")

import tensorflow as tf
import matplotlib.pyplot as plt

from image_classification import vgg16
from image_classification import utils


labels = []


img_path = input("Input the path and image name:")
img_ready = utils.load_image(img_path)

# 定义一个figure画图窗口，并制定窗口的名称，也可以设置窗口修的大小
fig = plt.figure("Top-5 预测结果")

with tf.Session() as sess:
    # 定义一个维度为【1,224,224,3】，类型为float32的tensor占位符
    x = tf.placeholder(tf.float32, [1, 224, 224, 3])
    vgg = vgg16.Vgg16()
    vgg.forward()

    # 将一个batch的数据喂入网络，得到网络的预测输出
    probability = sess.run(vgg.prob, feed_dict={x: img_ready})

    # np.argsort 函数返回预测值（probablity的数据结构[[各预测类别的概率值]]于小到大的索引值），并取出概率最大的五个索引值
    top5 = np.argsort(probability[0])[-1:-6:-1]
    print("top5:", top5)

    # 定义两个list -- 对应的概率值和实际标签
    values = []
    bar_label = []

    for n, i in enumerate(top5):
        print("n:", n)
        print("i:", i)

        values.append(probability[0][i])
        bar_label.append(labels[i])  # 根据索引值取出的实际标签放入bar_label

        print(i, ":", labels[i], "---", utils.percent(probability[0][i]))  # 打印属于某个类别的概率

    ax = fig.add_subplot(111)
    ax.bar(range(len(values)), values, tick_label=bar_label, width=0.5, fc="g")
    ax.set_ylabel("probability")
    ax.set_title("Top-5")

    for a, b in zip(range(len(values)), values):
        ax.text(a, b+0.0005, utils.percent(b), ha="center", va="bottom", fontsize=7)
    plt.savefig("./result.jpg")
    plt.show()