#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/3/19 12:28 PM
@File    : visualization.py
@Desc    : mnist数据集可视化

"""


import matplotlib.pyplot as plt
import numpy as np



def plot_images_labels_prediction(images,  # 图像列表
                                  labels,  # 标签列表
                                  prediction,  # 预测值列表
                                  index,  # 从第index个开始显示
                                  num=10):  # 一次显示多少幅
    fig = plt.gcf()  # 获取当前图表
    fig.set_size_inches(10, 12)
    if num > 25:
        num = 25

    for i in range(0, num):
        ax = plt.subplot(5, 5, i+1)
        ax.imshow(np.reshape(images[index], (28, 28)), cmap="binary")
        title = "label=" + str(np.argmax(labels[index]))
        if len(prediction) > 0:
            title += ", predict=" + str(prediction[index])
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        index += 1

    plt.show()
