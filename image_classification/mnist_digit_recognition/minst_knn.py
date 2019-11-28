#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author   : Joshua_Zou
@Contact  : joshua_zou@163.com
@Time     : 2018/3/30 22:11
@Software : PyCharm
@File     : minst_knn.py
@Desc     :用最近邻识别手写数字
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from minst.data_processing import *
import time


def knn(newinput,dataset,label,k):
    m = dataset.shape[0]
    diff = tile(newinput, (m, 1)) - dataset
    squreDiff = diff ** 2
    squreDist = sum(squreDiff, axis=1)
    distance = squreDist ** 0.5
    sortedDistIndices = argsort(distance)
    classCount = {}
    for i in range(k):
        voteLabel = label[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    maxCount = 0
    for k, v in classCount.items():
        if v > maxCount:
            maxCount = v
            maxIndex = k
    return maxIndex

def test_knn():
    ## step 1: load data
    print("step 1: load data...")
    train_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_img"
    train_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_lab\train_label.txt"
    images = img2mat(train_img_savefile, filenum=60000)
    labels = lab2mat(train_lab_savefile, rownum=60000, num_classes=10, one_hot=False)
    train = dataset(images, labels)
    test_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_img"
    test_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_lab\test_label.txt"
    t_images = img2mat(test_img_savefile, filenum=10000)
    t_labels = lab2mat(test_lab_savefile, rownum=10000, num_classes=10, one_hot=False)

    ## step 2: training...
    print("step 2: training...")
    pass

    ## step 3: testing
    print("step 3: testing...")
    numTestSamples = t_images.shape[0]
    matchCount = 0
    t_start = time.time()
    for i in range(numTestSamples):
        predict = knn(t_images[i], images, labels, 3)
        if predict == t_labels[i]:
            matchCount += 1
    accuracy = float(matchCount) / numTestSamples
    t_end = time.time()
    print("Running time:%" % (t_end - t_start))
    ## step 4: show the result
    print("step 4: show the result...")
    print('The classify accuracy is: %.2f%%' % (accuracy * 100))


if __name__ == "__main__":
    test_knn()