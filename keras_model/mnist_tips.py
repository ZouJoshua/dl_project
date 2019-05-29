#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-5-28 上午11:58
@File    : mnist_tips.py
@Desc    : 
"""

import numpy as np
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation


# 加载数据

def load_data():
    (x_train,y_train),(x_test,y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[:number] # 完整训练数据有6w,这里取前1w
    y_train = y_train[:number]
    x_train = x_train.reshape(number, 28*28)  # 原始数据是3维,这里变成2维
    x_test=x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)  # 原始数据是1,2...9这样的数字,to_categorical将其变成向量,对应的数字位置为1,其余为0
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train / 255
    x_test = x_test / 255
    return (x_train, y_train), (x_test, y_test)

(x_train,y_train),(x_test,y_test) = load_data()


# 选择合适的loss函数
def t_loss_mse():
    model = Sequential()
    model.add(Dense(input_dim=28*28, units=689, activation='sigmoid'))
    model.add(Dense(units=689, activation='sigmoid'))
    model.add(Dense(units=689, activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))  # 输出层10个节点
    model.compile(loss='mse', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    train_result = model.evaluate(x_train, y_train, batch_size=10000)
    test_result = model.evaluate(x_test, y_test, batch_size=10000)
    print('Train Accc:', train_result[1])
    print('Test Accc:', test_result[1])


def t_loss_categorical_crossentropy():
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=689, activation='sigmoid'))
    model.add(Dense(units=689, activation='sigmoid'))
    model.add(Dense(units=689, activation='sigmoid'))
    model.add(Dense(units=10, activation='softmax'))  # 输出层10个节点
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    train_result = model.evaluate(x_train, y_train, batch_size=10000)
    test_result = model.evaluate(x_test, y_test, batch_size=10000)
    print('Train Accc:', train_result[1])
    print('Test Accc:', test_result[1])


# 合适的隐藏层数量

def t_num_hid():
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=689, activation='sigmoid'))  # sigmoid会导致vanish gradient problem(梯度消失)
    for _ in range(10):
        model.add(Dense(units=689, activation='sigmoid'))  # 来个10层
    model.add(Dense(units=10, activation='softmax'))  # 输出层10个节点
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    train_result = model.evaluate(x_train, y_train, batch_size=10000)
    test_result = model.evaluate(x_test, y_test, batch_size=10000)
    print('Train Accc:', train_result[1])
    print('Test Accc:', test_result[1])


# 合适的激活函数

def t_activation_relu():
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=689, activation='relu'))
    for _ in range(10):
        model.add(Dense(units=689, activation='relu'))  # 来个10层
    model.add(Dense(units=10, activation='softmax'))  # 输出层10个节点
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    train_result = model.evaluate(x_train, y_train, batch_size=10000)
    test_result = model.evaluate(x_test, y_test, batch_size=10000)
    print('Train Accc:', train_result[1])
    print('Test Accc:', test_result[1])


# 合适的batch_size( batch_size 过大速度快,但会影响精度.而过小则速度会慢)
def t_batch_size():
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=689, activation='relu'))
    for _ in range(10):
        model.add(Dense(units=689, activation='relu'))  # 来个10层
    model.add(Dense(units=10, activation='softmax'))  # 输出层10个节点
    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=10000, epochs=20)

    train_result = model.evaluate(x_train, y_train, batch_size=10000)
    test_result = model.evaluate(x_test, y_test, batch_size=10000)
    print('Train Accc:', train_result[1])
    print('Test Accc:', test_result[1])


# 合适的optimizer
def t_optimizer_adam():
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=689, activation='relu'))
    for _ in range(10):
        model.add(Dense(units=689, activation='relu'))  # 来个10层
    model.add(Dense(units=10, activation='softmax'))  # 输出层10个节点
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    train_result = model.evaluate(x_train, y_train, batch_size=10000)
    test_result = model.evaluate(x_test, y_test, batch_size=10000)
    print('Train Accc:', train_result[1])
    print('Test Accc:', test_result[1])

# Dropout层

def load_data_normal():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    number = 10000
    x_train = x_train[:number]  # 完整训练数据有6w,这里取前1w
    y_train = y_train[:number]
    x_train = x_train.reshape(number, 28*28)  # 原始数据是3维,这里变成2维
    x_test=x_test.reshape(x_test.shape[0], 28*28)
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    y_train = np_utils.to_categorical(y_train, 10)  # 原始数据是1,2...9这样的数字,to_categorical将其变成向量,对应的数字位置为1,其余为0
    y_test = np_utils.to_categorical(y_test, 10)
    x_train = x_train / 255  # 特征工程
    x_test = x_test / 255
    x_test=np.random.normal(x_test)  # 加噪声
    return (x_train, y_train), (x_test, y_test)



def t_dropout():
    """2层relu+adam+categorical_crossentropy+batch_size=100"""
    (x_train, y_train), (x_test, y_test) = load_data_normal()
    model = Sequential()
    model.add(Dense(input_dim=28 * 28, units=689, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units=689, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units=689, activation='relu'))
    model.add(Dropout(0.7))
    model.add(Dense(units=10, activation='softmax'))  # 输出层10个节点
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=100, epochs=20)

    train_result = model.evaluate(x_train, y_train, batch_size=10000)
    test_result = model.evaluate(x_test, y_test, batch_size=10000)
    print('Train Accc:', train_result[1])
    print('Test Accc:', test_result[1])

if __name__ == "__main__":
    t_loss_mse()