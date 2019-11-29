#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/29/19 6:35 PM
@File    : vgg16.py
@Desc    : 还原网络和参数

"""


import inspect
import os
import numpy as np
import tensorflow as tf
import time
import matplotlib.pyplot as plt


VGG_MEAN = [103.939, 116.779, 123.68]  # 样本RGB的平均值


class Vgg16():

    def __init__(self, vgg16_path=None):
        if vgg16_path is None:
            vgg16_path = os.path.join(os.getcwd(), 'vgg16.npy')
            print(vgg16_path)
            self.data_dict = np.load(vgg16_path, encoding='latinl').items()  # 遍历其内键值对，导入模型参数

        for x in self.data_dict:
            print(x)

    def forward(self, images):
        """
        前向传播
        :param images:
        :return:
        """
        # plt.figure("process pictures")
        print("build model started")
        start_time = time.time()
        rgb_scaled = images*255.0  # 逐像素乘以255.0
        # 从GRB转换色彩通道到BGR，也可以使用cv中的GRBtoBGR
        red, green, blue = tf.split(rgb_scaled, 3, 3)
        assert red.get_shape().as_list()[1:] == [224, 224, 1]
        assert green.get_shape().as_list()[1:] == [224, 224, 1]
        assert blue.get_shape().as_list()[1:] == [224, 224, 1]
        # 逐样本减去每个通道的像素平均值，这种操作可以移除图像的平均亮度值，该方法常用在灰度图像上
        bgr = tf.concat([blue - VGG_MEAN[0], green - VGG_MEAN[1], red - VGG_MEAN[2]], 3)
        assert bgr.get_shape().as_list()[1:] == [224, 224, 3]

        # 构建VGG16层网络（包含5段卷积，3层全连接），并逐层根据命名空间读取网络参数
        # 第一段卷积，含有两个卷积层，后面接最大池化层，用来缩小图片尺寸
        # 传入命名空间的name， 来获取该层的卷积核和偏置，并做卷积运算， 最后返回经过激活函数后的值
        self.conv1_1 = self.convl_layer(bgr, "conv1_1")
        self.conv1_2 = self.convl_layer(self.conv1_1, "conv1_2")
        # 根据传入的pooling名字对该层做相应的池化操作
        self.pool1 = self.max_pool_2x2(self.conv1_2, "pool1")

        # 第二段卷积，同样包含两个卷积层， 一个最大池化层
        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool_2x2(self.conv2_2, "pool2")

        # 第三段卷积，包含三个卷积层，一个最大池化层
        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool_2x2(self.conv3_3, "pool3")

        # 第四段卷积，包含三个卷积层，一个最大池化层
        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool_2x2(self.conv4_3, "pool4")

        # 第五段卷积，包含三个卷积层，一个最大池化层
        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool_2x2(self.conv5_3, "pool5")

        # 第六层全连接
        self.fc6 = self.fc_layer(self.pool5, "fc6")  # 根据命名空间name做加权求和运算
        assert self.fc6.get_shape().as_list()[1:] == [4096]  # 4096是该层输出后的长度
        self.relu6 = tf.nn.relu(self.fc6)

        # 第七层全连接
        self.fc7 = self.fc_layer(self.relu6, "fc7")
        self.relu7 = tf.nn.relu(self.fc7)

        # 第八层全连接
        # 经过最后一层的全连接后，再做softmax分类
        self.fc8 = self.fc_layer(self.relu7, "fc8")
        self.prob = tf.nn.softmax(self.fc8, name="prob")

        end_time = time.time()
        print("time consuming: %f" % (end_time - start_time))

        self.data_dict = None  # 清空本次读取到的模型参数字典

    def conv_layer(self, x, name):
        """
        定义卷积运算
        :param x:
        :param name:
        :return:
        """
        with tf.variable_scope(name):  # 根据命名空间找到对应卷积层的网络参数
            w = self.get_conv_filter(name)  # 读取到该层的卷积核
            conv = tf.nn.conv2d(x, w, [1, 1, 1, 1], padding="SAME")  # 卷积运算
            conv_biases = self.get_bias(name)  # 读取偏置项
            result = tf.nn.relu(tf.nn.bias_add(conv, conv_biases))  # 加上偏置，并做激活计算
            return result

    def get_conv_filter(self, name):
        """
        定义获取卷积核的函数
        :param name:
        :return:
        """
        # 根据命名空间name从参数字典中获取对应的卷积核
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        """
        定义获取偏置项的函数
        :param name:
        :return:
        """
        # 根据命名空间name从参数字典中获取对应的卷积核
        return tf.constant(self.data_dict[name][1], name="biases")


    def max_pool_2x2(self, x, name):
        """
        定义最大池化操作
        :param x:
        :param name:
        :return:
        """
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME", name=name)


    def fc_layer(self, x, name):
        """
        定义全连接层的前向传播计算
        :param x:
        :param name:
        :return:
        """
        with tf.variable_scope(name):  # 根据命名空间name做全连接计算
            shape = x.get_shape().as_list()  # 获取该层的维度信息列表
            print("fc_layer shape ", shape)
            dim = 1
            for i in shape[1:]:
                dim *= i  # 将每层的维度相乘
            # 改变特征图的形状，也就是将得到的多为特征做拉伸操作，只在进入第六层全连接操作时做该操作
            x = tf.reshape(x, [-1, dim])

            w = self.get_fc_weight(name)  # 读取权重值
            b = self.get_bias(name)  # 读取偏置项

            result = tf.nn.bias_add(tf.matmul(x, w), b)
            return result

    def get_fc_weight(self, name):
        """
        定义获取权重的函数
        :param name:
        :return:
        """
        return tf.constant(self.data_dict[name][0], name="weights")


