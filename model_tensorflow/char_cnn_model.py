#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/31/20 5:19 PM
@File    : char_cnn_model.py
@Desc    : 

"""


import tensorflow as tf
from model_tensorflow.basic_model import BaseModel
from math import sqrt
from model_tensorflow.basic_config import ConfigBase


class Config(ConfigBase):
    """charcnn配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)

        self.is_training = self.config.getboolean("is_training", False)                     # 是否训练
        self.large_params = self.config.getboolean("large_params", False)                   # char_cnn模型是否使用大参数
        self.conv_layers_size = eval(self.config.get("conv_layers_size", "[[256, 7, 3],\
                                                                        [256, 7, 3],\
                                                                        [256, 3, None],\
                                                                        [256, 3, None],\
                                                                        [256, 3, None],\
                                                                        [256, 3, 3]]"))     # 卷积层尺寸, a list of int. e.g.
        self.fc_layers_size = eval(self.config.get("fc_layers_size", "[1024,1024,1024]"))   # 全连接层神经元尺寸, a list of int





class CharCNN(BaseModel):
    """
    6层卷积层 + 3层全连接层
    针对不同大小的数据集,提出两种结构参数
    卷积层
    -------------------------------------------------------
    layer | large feature | small feature | kernel | pool
    -------------------------------------------------------
    1          1024            256            7       3
    2          1024            256            7       3
    3          1024            256            3      N/A
    4          1024            256            3      N/A
    5          1024            256            3      N/A
    6          1024            256            3       3
    -------------------------------------------------------

    全连接层
    -------------------------------------------------
    layer | output units large | output units small
    -------------------------------------------------
    7              2048                 1024
    8              2048                 1024
    9                depends on the problem
    _________________________________________________
    """

    def __init__(self, config, vocab_size, word_vectors):
        super(CharCNN, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        self.inputs = tf.placeholder(tf.int32, [None, config.sequence_length], name="inputs")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()


    def build_model(self):
        self.embedding_layer()
        self.conv_maxpool_fc_layer()
        # self.multi_conv_maxpool_layer()
        self.full_connection_layer()
        self.cal_loss()

        self.predictions = self.get_predictions()
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()

    def embedding_layer(self):
        """
        词嵌入层
        :return:
        """
        with tf.name_scope("embedding-layer"):
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config.embedding_dim],
                                          initializer=tf.contrib.layers.xavier_initializer())

            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，
            # 维度[batch_size, sequence_length, embedding_dim]
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)

            # 卷积操作conv2d的输入是四维[batch_size, sequence_length, embedding_dim, channel],
            # 分别代表着批处理大小、宽度、高度、通道数,因此需要增加维度,设为1,用tf.expand_dims来增大维度
            self.embedded_words_expand = tf.expand_dims(embedded_words, -1)


    def get_layers_size(self):
        """

        :return:
        """
        # 该列表中子列表的三个元素分别是卷积核的数量，卷积核的高度，池化的尺寸
        if not self.config.conv_layers_size and not self.config.fc_layers_size:
            if self.config.large_params:
                conv_num_filter, fc_num_filter = (1024, 2048)
            else:
                conv_num_filter, fc_num_filter = (256, 1024)
            self.conv_layers_size = [[conv_num_filter, 7, 3],
                                [conv_num_filter, 7, 3],
                                [conv_num_filter, 3, None],
                                [conv_num_filter, 3, None],
                                [conv_num_filter, 3, None],
                                [conv_num_filter, 3, 3]]
            if not self.config.hidden_size:
                output_size = fc_num_filter
            else:
                output_size = self.config.hidden_size

            self.fc_layers_size = [fc_num_filter, fc_num_filter, output_size]
        else:
            self.conv_layers_size = self.config.conv_layers_size
            self.fc_layers_size = self.config.fc_layers_size

    def conv_maxpool_fc_layer(self):
        """
        创建卷积层-池化层-全连接层
        # step1.create filters
        # step2.conv (CNN->BN->relu)
        # step3.apply nolinearity
        # step4.max-pooling
        # step5.fc-layer
        :return:
        """

        self.get_layers_size()

        for i, cl in enumerate(self.conv_layers_size):
            # print("开始第" + str(i + 1) + "卷积层的处理")
            # 利用命名空间name_scope来实现变量名复用
            with tf.name_scope("conv-layer-{}".format(i + 1)):

                # 获取字符的向量长度
                filter_dim = self.embedded_words_expand.get_shape()[2].value
                # filterShape = [height, width, in_channels, out_channels]
                filter_shape = [cl[1], filter_dim, 1, cl[0]]

                stdv = 1 / sqrt(cl[0] * cl[1])

                # 初始化w和b的值
                conv_w = tf.Variable(tf.random_uniform(filter_shape, minval=-stdv, maxval=stdv),
                                    dtype='float32', name="filter-{}".format(cl[0]))
                conv_b = tf.Variable(tf.random_uniform(shape=[cl[0]], minval=-stdv, maxval=stdv), name="filter-{}-b".format(cl[0]))

                # w_conv = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.05), name="w")
                # b_conv = tf.Variable(tf.constant(0.1, shape=[cl[0]]), name="b")
                # 构建卷积层，可以直接将卷积核的初始化方法传入（w_conv）
                conv = tf.nn.conv2d(
                    self.embedded_words_expand,
                    conv_w,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv-{}".format(i + 1))

                # print(conv.shape)
                # 加上偏差,可以直接加上relu函数,
                # 因为tf.nn.conv2d事实上是做了一个卷积运算，然后在这个运算结果上加上偏差，再导入到relu函数中
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")

                # with tf.name_scope("batchNormalization"):
                #     h = self._batchNorm(h)

                if cl[-1] is not None:
                    ksize_shape = [1, cl[2], 1, 1]
                    h_pool = tf.nn.max_pool(h, ksize=ksize_shape, strides=ksize_shape, padding="VALID", name="pool")
                else:
                    h_pool = h

                # print(h_pool.shape)

                # 对维度进行转换，转换成卷积层的输入维度
                self.embedded_words_expand = tf.transpose(h_pool, [0, 1, 3, 2], name="transpose")
            # print(self.embedded_words_expand.shape)
            # print(self.embedded_words_expand.get_shape())


        with tf.name_scope("reshape"):
            fc_dim = self.embedded_words_expand.get_shape()[1].value * self.embedded_words_expand.get_shape()[2].value
            self.pool_flat_output = tf.reshape(self.embedded_words_expand, [-1, fc_dim])

        self.weights = [fc_dim] + self.fc_layers_size

        for i, fl in enumerate(self.config.fc_layers_size):
            with tf.name_scope("fc-layer-%s" % (i + 1)):
                # print("开始第" + str(i + 1) + "全连接层的处理")
                stdv = 1 / sqrt(self.weights[i])

                # 定义全连接层的初始化方法，均匀分布初始化w和b的值
                fc_w = tf.Variable(tf.random_uniform([self.weights[i], fl], minval=-stdv, maxval=stdv), dtype="float32", name="w")
                fc_b = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv), dtype="float32", name="b")

                # fc_w = tf.Variable(tf.truncated_normal([weights[i], fl], stddev=0.05), name="W")
                # fc_b = tf.Variable(tf.constant(0.1, shape=[fl]), name="b")

                self.fc_layer_output = tf.nn.relu(tf.matmul(self.pool_flat_output, fc_w) + fc_b)

                with tf.name_scope("fc-layer-dropout"):
                    self.fc_layer_dropout = tf.nn.dropout(self.fc_layer_output, self.keep_prob)

            self.pool_flat_output = self.fc_layer_dropout


    def full_connection_layer(self):
        """
        全连接层，后面接dropout以及relu激活
        :return:
        """

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.pool_flat_output, self.keep_prob)

        # 定义隐层到输出层的权重系数和偏差的初始化方法
        # stdv = 1 / sqrt(self.weights[-1])
        # output_w = tf.Variable(tf.random_uniform([self.fc_layers_size[-1], self.config.num_labels], minval=-stdv, maxval=stdv), dtype="float32", name="w")
        # output_b = tf.Variable(tf.random_uniform(shape=[self.config.num_labels], minval=-stdv, maxval=stdv), name="b")

        # 全连接层的输出
        with tf.name_scope("fully_connection_layer"):
            output_w = tf.get_variable(
                "output_w",
                shape=[self.weights[-1], self.config.num_labels],
                initializer=tf.contrib.layers.xavier_initializer())
            output_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_labels]), name="output_b")
            self.logits = tf.nn.xw_plus_b(h_drop, output_w, output_b, name="logits")
            # self.logits = tf.matmul(h_drop, output_w) + output_b
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)

    def cal_loss(self):
        with tf.name_scope("loss"):
            # 计算交叉熵损失
            self.labels = tf.cast(self.labels, dtype=tf.int32)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.labels, logits=self.logits)
            loss = tf.reduce_mean(losses)

            # self.l2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * self.config.l2_reg_lambda
            # self.loss = loss + self.l2_loss
            self.loss = loss + self.config.l2_reg_lambda * self.l2_loss