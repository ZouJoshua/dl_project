#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@Author   : Joshua_Zou
@Contact  : joshua_zou@163.com
@Time     : 2018/3/12 13:40
@Software : PyCharm
@File     : data_processing.py
@Desc     :数据处理
"""

from PIL import Image
from pylab import *
import struct
import matplotlib.pyplot as plt
import os
import shutil
import numpy as np


def read_image(filename, savefilepath):
    """
    解析并保存图片
    :param filename:源文件图片
    :param savefilepath:保存路径
    :return:图片
    """
    # 使用二进制方式读取文件
    f = open(filename, 'rb')
    buf = f.read()
    f.close()

    # 使用大端法读取4个unsinged int32，使用struc.unpack_from （'>IIII'）
    index = 0
    magic, images, rows, columns = struct.unpack_from('>IIII', buf, index)
    index += struct.calcsize('>IIII')

    for i in range(images):
        # for i in range(2000):
        # 创建一张空白的图片，L代表灰度图，逐个像素读取
        image = Image.new('L', (columns, rows))
        for x in range(rows):
            for y in range(columns):
                image.putpixel((y, x), int(struct.unpack_from('>B', buf, index)[0]))
                index += struct.calcsize('>B')
            image.save(savefilepath + '\\' + str(i) + '.png')
        if (i + 1) % 10000 == 0:
            print('已保存到第 ' + str(i) + 'image')


def read_label(filename, saveFilename):
    """
    解析并将标签保存为txt文件
    :param filename:二进制标签文件
    :param saveFilename:保存路径
    :return:txt标签文件
    """
    f = open(filename, 'rb')
    buf = f.read()
    f.close()

    index = 0
    magic, labels = struct.unpack_from('>II', buf, index)
    index += struct.calcsize('>II')

    labelArr = [0] * labels
    # labelArr = [0] * 2000

    for x in range(labels):
        # for x in range(100):
        labelArr[x] = int(struct.unpack_from('>B', buf, index)[0])
        index += struct.calcsize('>B')
    save = open(saveFilename, 'w')
    save.write(','.join(map(lambda x: str(x), labelArr)))
    save.write('\n')
    save.close()
    print('save labels success')


def get_imgdata(idx3_file):
    """
    将图片的二进制文件直接解析为数据
    :param idx3_file: 图片的二进制源文件
    :return:返回所有图片的像素重构的矩阵
    """
    with open(idx3_file, 'rb') as file:
        bin_data = file.read()

    # 解析文件头信息，依次为魔数、图片数量、每张图片高、每张图片宽
    offset = 0
    fmt_header = '>iiii'  # '>IIII'是说使用大端法读取4个unsinged int32
    magic_number, num_images, num_rows, num_cols = struct.unpack_from(fmt_header, bin_data, offset)
    # print ('魔数:%d, 图片数量: %d张, 图片大小: %d*%d' % (magic_number, num_images, num_rows, num_cols))

    # 解析数据集
    image_size = num_rows * num_cols
    offset += struct.calcsize(fmt_header)
    # print("offset: ",offset)
    fmt_image = '>' + str(image_size) + 'B'  # '>784B'的意思就是用大端法读取784个unsigned byte
    images = np.empty((num_images, num_rows * num_cols))
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        images[i] = np.array(struct.unpack_from(fmt_image, bin_data, offset)).reshape((num_rows * num_cols))
        offset += struct.calcsize(fmt_image)
    return images


def get_imglabel(idx1_file):
    """
    将标签的二进制文件解析为数据矩阵
    :param idx1_file: 标签的二进制文件
    :return: 返回训练标签矩阵
    """
    with open(idx1_file, 'rb') as file:
        bin_data = file.read()
        # 解析文件头信息，依次为魔数和标签数
    offset = 0
    fmt_header = '>ii'
    magic_number, num_images = struct.unpack_from(fmt_header, bin_data, offset)
    print('魔数:%d, 图片数量: %d张' % (magic_number, num_images))

    # 解析数据集
    offset += struct.calcsize(fmt_header)
    fmt_image = '>B'
    labels = np.empty(num_images)
    for i in range(num_images):
        if (i + 1) % 10000 == 0:
            print('已解析 %d' % (i + 1) + '张')
        labels[i] = struct.unpack_from(fmt_image, bin_data, offset)[0]
        offset += struct.calcsize(fmt_image)
    return labels


def img2mat(filepath, filenum):
    """
    逐张处理图片为数据矩阵
    :param filepath:
    :param filenum:
    :return:
    """
    images = np.zeros((filenum, 28 * 28))
    for i in range(filenum):
        tmp_file = filepath + '\\' + str(i) + '.png'
        im = array(Image.open(tmp_file).convert("L"))
        im = np.multiply(im, 1.0 / 255.0).reshape(1, 28 * 28)
        images[i, :] = im
    return images


def lab2mat(file, rownum,num_classes,one_hot = False):
    with open(file, "r") as f:
        lab = f.read()
        lab_tmp = lab.rstrip("\n").split(",")
        labels = np.array([int(i) for i in lab_tmp]).reshape(rownum, 1)
    if one_hot:
        return dense_to_one_hot(labels, num_classes)
    return labels

def dense_to_one_hot(labels_dense, num_classes):
    """Convert class labels from scalars to one-hot vectors."""
    num_labels = labels_dense.shape[0]
    index_offset = np.arange(num_labels) * num_classes
    labels_one_hot = np.zeros((num_labels, num_classes))
    labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
    return labels_one_hot

class dataset(object):
    def __init__(self,images,labels):
        self._epochs_completed = 0
        self._index_in_epoch = 0
        self._images = images
        self._labels = labels
        self._num_examples = 60000

    @property
    def images(self):
        return self._images

    @property
    def labels(self):
        return self._labels

    @property
    def num_examples(self):
        return self._num_examples

    @property
    def epochs_completed(self):
        return self._epochs_completed

    def next_batch(self,batch_size):
        start = self._index_in_epoch
        self._index_in_epoch += batch_size
        if self._index_in_epoch > self._num_examples:
            # Finished epoch
            self._epochs_completed += 1
            # Shuffle the data
            perm = np.arange(self._num_examples)
            np.random.shuffle(perm)
            self._images = self._images[perm]
            self._labels = self._labels[perm]
            # Start next epoch
            start = 0
            self._index_in_epoch = batch_size
            assert batch_size <= self._num_examples
        end = self._index_in_epoch
        return self._images[start:end],self._labels[start:end]




if __name__  == '__main__':
    # train_img_file = r"D:\Python\Anaconda\TensorFlow\file\minst\train-images.idx3-ubyte"
    train_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_img"
    # train_lab_file = r"D:\Python\Anaconda\TensorFlow\file\minst\train-labels.idx1-ubyte"
    train_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\train_lab\train_label.txt"
    # test_img_file = r"D:\Python\Anaconda\TensorFlow\file\minst\t10k-images.idx3-ubyte"
    # test_img_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_img"
    # test_lab_file = r"D:\Python\Anaconda\TensorFlow\file\minst\t10k-labels.idx1-ubyte"
    # test_lab_savefile = r"D:\Python\Anaconda\TensorFlow\file\minst\test_lab\test_label.txt"

    # read_image(train_img_file,train_img_savefile)
    # read_image(test_img_file,test_img_savefile)
    # read_label(train_lab_file,train_lab_savefile)
    # read_label(test_lab_file,test_lab_savefile)
    images = img2mat(train_img_savefile,1)
    print(images[0])
    labels = lab2mat(train_lab_savefile,rownum = 60000,num_classes=10,one_hot=True)
    print(labels[:10])


