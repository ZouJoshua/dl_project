#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:33
@File    : textcnn_model.py
@Desc    : TextCNN

"""


import tensorflow as tf
from model_tensorflow.basic_model import BaseModel
from model_tensorflow.basic_config import ConfigBase

class Config(ConfigBase):
    """textcnn配置参数"""
    def __init__(self, config_file, section):
        super(Config, self).__init__(config_file, section=section)

        self.filter_sizes = eval(self.config.get("filter_sizes", "[3,4,5]"))       # 卷积核尺寸, a list of int. e.g. [3,4,5]
        self.num_filters = self.config.getint("num_filters")                      # 卷积核数量(channels数)
        self.is_training = self.config.getboolean("is_training", True)            # 是否训练




class TextCNN(BaseModel):

    def __init__(self, config, vocab_size, word_vectors):
        super(TextCNN, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        self.num_filters_total = self.config.num_filters * len(self.config.filter_sizes)
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()



    def build_model(self):
        self.embedding_layer()
        self.conv_maxpool_layer()
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

    def conv_maxpool_layer(self):
        """
        创建卷积和池化层
        # step1.create filters
        # step2.conv (CNN->BN->relu)
        # step3.apply nolinearity
        # step4.max-pooling
        :return:
        """
        #
        pooled_outputs = []
        # 有三种size的filter，3， 4， 5，textCNN是个多通道单层卷积的模型，可以看作三个单层的卷积模型的融合
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv-maxpool-{}".format(filter_size)):
                # 卷积层
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                conv_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="filter-{}".format(filter_size))
                conv_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="filter-{}-b".format(filter_size))
                conv = tf.nn.conv2d(
                    self.embedded_words_expand,
                    conv_filter,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv-{}".format(filter_size))  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training, scope='cnn_bn_')

                # relu函数的非线性映射
                # 卷积层的输出h，即每个feature map
                # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), name="relu")
                # 池化层
                # 最大池化，池化是对卷积后的序列取一个最大值,本质上是一个特征向量，最后一个维度是特征代表数量
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, self.config.sequence_length - filter_size + 1, 1, 1],  # ksize shape: [batch, height, width, channels],一般为[1,height,width,1]，batch和channels上不池化
                    strides=[1, 1, 1, 1],
                    padding='VALID',
                    name="pool")  # shape: [batch_size, 1, 1, num_filters]

                pooled_outputs.append(pooled)  # 将三种size的filter的输出一起加入到列表中

        # 特征concat
        # 池化后的维度不变，按照最后的维度channel来concat
        # 把每一个max-pooling之后的张量合并起来之后得到一个长向量
        h_pool = tf.concat(pooled_outputs, 3)  # shape: [batch_size, 1, 1, num_filters_total]

        # 摊平成二维的数据输入到全连接层
        self.pool_flat_output = tf.reshape(h_pool, [-1, self.num_filters_total])  # shape: [batch_size, num_filters_total]


    def multi_conv_maxpool_layer(self):
        """
        创建卷积和池化层
        # step1.create filters
        # step2.conv (CNN->BN->relu)
        # step3.apply nolinearity
        # step4.max-pooling
        :return:
        """
        pooled_outputs = []
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.variable_scope("conv-maxpool-{}".format(filter_size)):
                # Layer1:
                # 1) CNN->BN->relu
                filter_shape = [filter_size, self.config.embedding_dim, 1, self.config.num_filters]
                conv_filter = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="conv-filter-{}".format(filter_size))
                conv_b = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="conv-filter-{}-b".format(filter_size))

                conv = tf.nn.conv2d(
                    self.embedded_words_expand,
                    conv_filter,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.config.is_training, scope='cnn1')
                h = tf.nn.relu(tf.nn.bias_add(conv, conv_b), "relu")  # shape: [batch_size,sequence_length,1,num_filters]
                h = tf.reshape(h, [-1, self.config.sequence_length, self.config.num_filters, 1])  # shape: [batch_size,sequence_length,num_filters,1]


                # Layer2:
                # 2) CNN->BN->relu
                filter2_shape = [filter_size, self.config.num_filters, 1, self.config.num_filters]
                conv_filter2 = tf.Variable(tf.truncated_normal(filter2_shape, stddev=0.1), name="conv2-filter-{}".format(filter_size))
                conv_b2 = tf.Variable(tf.constant(0.1, shape=[self.config.num_filters]), name="conv2-filter-{}-b".format(filter_size))

                conv2 = tf.nn.conv2d(
                    h,
                    conv_filter2,
                    strides=[1, 1, 1, 1],
                    padding="SAME",
                    name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.config.is_training, scope='cnn2')
                h = tf.nn.relu(tf.nn.bias_add(conv2, conv_b2), "relu2")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # Max-pooling
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, self.config.sequence_length, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1))     # [batch_size,num_filters]
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1)  # [batch_size,num_filters*2]
                pooled_outputs.append(pooling_max)  # [batch_size,num_filters]
        # concat
        self.pool_flat_output = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_filters_total]


    def full_connection_layer(self):
        """
        全连接层，后面接dropout以及relu激活
        :return:
        """

        # dropout
        with tf.name_scope("dropout"):
            h_drop = tf.nn.dropout(self.pool_flat_output, self.keep_prob)


        # 全连接层的输出
        with tf.name_scope("fully_connection_layer"):
            output_w = tf.get_variable(
                "output_w",
                shape=[self.num_filters_total, self.config.num_labels],
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

