#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 1/2/20 10:19 PM
@File    : text_category_dl.py
@Desc    : 文本分类(深度学习)

"""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import numpy as np
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score, precision_score, classification_report
import tensorflow as tf
from tensorflow.contrib.layers.python.layers import encoders

import pickle
from nlp_tasks.text_classification.brief_news.preprocess import DataSet, dump_data
from setting import DATA_PATH

# import os
# os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

learn = tf.contrib.learn

FLAGS = None

# MAX_DOCUMENT_LENGTH = 15
# MIN_WORD_FREQUENCE = 1
# EMBEDDING_SIZE = 50

# 文档最长长度
MAX_DOCUMENT_LENGTH = 100
# 最小词频数
MIN_WORD_FREQUENCE = 2
# 词嵌入的维度
EMBEDDING_SIZE = 20
# filter个数
N_FILTERS = 10
# 感知野大小
WINDOW_SIZE = 20
# filter的形状
FILTER_SHAPE1 = [WINDOW_SIZE, EMBEDDING_SIZE]
FILTER_SHAPE2 = [WINDOW_SIZE, N_FILTERS]
# 池化
POOLING_WINDOW = 4
POOLING_STRIDE = 2
n_words = 0


def cnn_model(features, target):
    """
    2层的卷积神经网络，用于短文本分类
    :param features:
    :param target:
    :return:
    """
    # 先把词转成词嵌入
    # 我们得到一个形状为[n_words, EMBEDDING_SIZE]的词表映射矩阵
    #  接着我们可以把一批文本映射成[batch_size, sequence_length, EMBEDDING_SIZE]的矩阵形式
    target = tf.one_hot(target, 6, 1, 0)
    word_vectors = tf.contrib.layers.embed_sequence(features,
                                                    vocab_size=n_words,
                                                    embed_dim=EMBEDDING_SIZE,
                                                    scope='words')
    word_vectors = tf.expand_dims(word_vectors, 3)
    with tf.variable_scope('CNN_Layer1'):
        # 添加卷积层做滤波
        conv1 = tf.contrib.layers.conv2d(word_vectors, N_FILTERS, FILTER_SHAPE1, padding='VALID')
        # 添加RELU非线性
        conv1 = tf.nn.relu(conv1)
        # 最大池化
        pool1 = tf.nn.max_pool(conv1,
                               ksize=[1, POOLING_WINDOW, 1, 1],
                               strides=[1, POOLING_STRIDE, 1, 1],
                               padding='SAME')
        # 对矩阵进行转置，以满足形状
        pool1 = tf.transpose(pool1, [0, 1, 3, 2])
    with tf.variable_scope('CNN_Layer2'):
        # 第2个卷积层
        conv2 = tf.contrib.layers.conv2d(pool1,
                                                N_FILTERS,
                                                FILTER_SHAPE2,
                                                padding='VALID')
        # 抽取特征
        pool2 = tf.squeeze(tf.reduce_max(conv2, 1), squeeze_dims=[1])

    # 全连接层
    logits = tf.contrib.layers.fully_connected(pool2, 6, activation_fn=None)
    loss = tf.losses.softmax_cross_entropy(target, logits)

    train_op = tf.contrib.layers.optimize_loss(loss,
                                               tf.contrib.framework.get_global_step(),
                                               optimizer='Adam',
                                               learning_rate=0.001)
    return {'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_op



def rnn_model(features, target):
    """用RNN模型(这里用的是GRU)完成文本分类"""
    # Convert indexes of words into embeddings.
    # This creates embeddings matrix of [n_words, EMBEDDING_SIZE] and then
    # maps word indexes of the sequence into [batch_size, sequence_length, EMBEDDING_SIZE].
    word_vectors = tf.contrib.layers.embed_sequence(
        features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE, scope='words')
    # Split into list of embedding per word, while removing doc length dim.
    # word_list results to be a list of tensors [batch_size, EMBEDDING_SIZE].
    word_list = tf.unstack(word_vectors, axis=1)

    # Create a Gated Recurrent Unit cell with hidden size of EMBEDDING_SIZE.
    cell = tf.contrib.rnn.GRUCell(EMBEDDING_SIZE)

    # Create an unrolled Recurrent Neural Networks to length of
    # MAX_DOCUMENT_LENGTH and passes word_list as inputs for each unit.
    _, encoding = tf.contrib.rnn.static_rnn(cell, word_list, dtype=tf.float32)

    # Given encoding of RNN, take encoding of last step (e.g hidden size of the
    # neural network of last step) and pass it as features for logistic regression over output classes.
    target = tf.one_hot(target, 6, 1, 0)
    logits = tf.contrib.layers.fully_connected(encoding, 6, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)

    # Create a training op.
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)

    return {'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_op


def bag_of_words_model(features, target):
    """先转成词袋模型"""
    target = tf.one_hot(target, 6, 1, 0)
    features = encoders.bow_encoder(
        features, vocab_size=n_words, embed_dim=EMBEDDING_SIZE)
    logits = tf.contrib.layers.fully_connected(features, 6, activation_fn=None)
    loss = tf.contrib.losses.softmax_cross_entropy(logits, target)
    train_op = tf.contrib.layers.optimize_loss(
        loss,
        tf.contrib.framework.get_global_step(),
        optimizer='Adam',
        learning_rate=0.01)
    return {'class': tf.argmax(logits, 1), 'prob': tf.nn.softmax(logits)}, loss, train_op


def train_cnn_model():
    global n_words
    # 处理词汇
    # cat_list = ["car"]
    cat_list = ["car", "entertainment", "finance", "sports", "military", "technology"]

    data_path = os.path.join(DATA_PATH, "brief_news")
    train_dump_file = os.path.join(data_path, "train.pkl")
    test_dump_file = os.path.join(data_path, "test.pkl")
    if os.path.exists(train_dump_file) and os.path.exists(test_dump_file):
        with open(train_dump_file, 'rb') as file_train, open(test_dump_file, "rb") as file_test:
            train_data = pickle.load(file_train)
            test_data = pickle.load(file_test)
    else:
        ds = DataSet(DATA_PATH, cat_list)
        # x, y = zip(*ds.data)
        train_data = (ds.x_train, ds.y_train)
        test_data = (ds.x_test, ds.y_test)
        dump_data(train_data, train_dump_file)
        dump_data(test_data, test_dump_file)

    x_train, y_train = train_data
    x_test, y_test = test_data

    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    cate_dic = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5, 'finance': 6}
    train_target = list(map(lambda x: cate_dic[x], y_train))
    test_target = list(map(lambda x: cate_dic[x], y_test))
    # print(x_train)
    # print(train_target)
    y_train = pd.Series(train_target)
    y_test = pd.Series(test_target)

    # 构建模型
    classifier = learn.SKCompat(learn.Estimator(model_fn=cnn_model))

    # 训练和预测
    classifier.fit(x_train, y_train, steps=100)
    y_predicted = classifier.predict(x_test)['class']
    print(y_predicted)
    score = metrics.accuracy_score(y_test, y_predicted)
    class_report = classification_report(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))
    print("classification report\n{}".format(class_report))


def train_rnn_model():
    global n_words
    # 处理词汇
    cat_list = ["car", "entertainment", "finance", "sports", "military", "technology"]
    data_path = os.path.join(DATA_PATH, "brief_news")
    train_dump_file = os.path.join(data_path, "train.pkl")
    test_dump_file = os.path.join(data_path, "test.pkl")
    if os.path.exists(train_dump_file) and os.path.exists(test_dump_file):
        with open(train_dump_file, 'rb') as file_train, open(test_dump_file, "rb") as file_test:
            train_data = pickle.load(file_train)
            test_data = pickle.load(file_test)
    else:
        ds = DataSet(DATA_PATH, cat_list)
        # x, y = zip(*ds.data)
        train_data = (ds.x_train, ds.y_train)
        test_data = (ds.x_test, ds.y_test)
        dump_data(train_data, train_dump_file)
        dump_data(test_data, test_dump_file)

    x_train, y_train = train_data
    x_test, y_test = test_data

    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    cate_dic = {'technology': 1, 'car': 2, 'entertainment': 3, 'military': 4, 'sports': 5, 'finance': 6}
    train_target = list(map(lambda x: cate_dic[x], y_train))
    test_target = list(map(lambda x: cate_dic[x], y_test))
    y_train = pd.Series(train_target)
    y_test = pd.Series(test_target)
    model_fn = rnn_model
    classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn))

    # Train and predict
    classifier.fit(x_train, y_train, steps=100)
    y_predicted = classifier.predict(x_test)['class']

    score = metrics.accuracy_score(y_test, y_predicted)
    class_report = classification_report(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))
    print("classification report\n{}".format(class_report))

def train_bag_of_word_model():
    global n_words
    # 处理词汇
    cat_list = ["car", "entertainment", "finance", "sports", "military", "technology"]
    data_path = os.path.join(DATA_PATH, "brief_news")
    train_dump_file = os.path.join(data_path, "train.pkl")
    test_dump_file = os.path.join(data_path, "test.pkl")

    if os.path.exists(train_dump_file) and os.path.exists(test_dump_file):
        with open(train_dump_file, 'rb') as file_train, open(test_dump_file, "rb") as file_test:
            train_data = pickle.load(file_train)
            test_data = pickle.load(file_test)
    else:
        ds = DataSet(DATA_PATH, cat_list)
        # x, y = zip(*ds.data)
        train_data = (ds.x_train, ds.y_train)
        test_data = (ds.x_test, ds.y_test)
        dump_data(train_data, train_dump_file)
        dump_data(test_data, test_dump_file)

    x_train, y_train = train_data
    x_test, y_test = test_data

    # 处理词汇
    vocab_processor = learn.preprocessing.VocabularyProcessor(MAX_DOCUMENT_LENGTH, min_frequency=MIN_WORD_FREQUENCE)
    x_train = np.array(list(vocab_processor.fit_transform(x_train)))
    x_test = np.array(list(vocab_processor.transform(x_test)))
    n_words = len(vocab_processor.vocabulary_)
    print('Total words: %d' % n_words)

    model_fn = bag_of_words_model
    classifier = learn.SKCompat(learn.Estimator(model_fn=model_fn))

    # Train and predict
    classifier.fit(x_train, y_train, steps=1000)
    y_predicted = classifier.predict(x_test)['class']
    score = metrics.accuracy_score(y_test, y_predicted)
    print('Accuracy: {0:f}'.format(score))


if __name__ == "__main__":
    train_cnn_model()
    # train_rnn_model()
    # train_bag_of_word_model()