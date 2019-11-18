#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/17/19 4:33 PM
@File    : tf2_4.py
@Desc    : 优化参数（滑动平均值）

"""

"""
滑动平均值，记录每个参数一段时间内过往值的平均，增加模型的泛化性，针对参数w和b

滑动平均值= 衰减率×滑动平均值 + （1-衰减率）×参数
衰减率= min{MOVING_AVERAGE_DECAY, （1+轮数）/（10+轮数）}
"""

import tensorflow as tf



# 定义变量和滑动平均
# 定义一个32浮点变量，初始值为0.0，这个代码是不断更新w1参数，优化w1参数，滑动平均做了w1的影子

w1 = tf.Variable(0, dtype=tf.float32)
# 定义num_updates（迭代轮数），初始值为0，不可被训练
global_step = tf.Variable(0, trainable=False)
# 实例化滑动平均类，衰减率为0.99，当前轮数global_step
MOVING_AVERAGE_DECAY = 0.99
ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
# ema.apply后的括号里是更新列表，每次运行sess.run（ema_op）时，对更新列表里的元素求滑动平均值
# 在实际应用中，使用tf.trainable_variables()自动讲所有待训练的参数汇总为列表
# ema_op = ema.apply([w1])
ema_op = ema.apply(tf.trainable_variables())


# 查看不懂迭代中变量取值的变化

with tf.Session() as sess:
    # 初始化
    init_op = tf.global_variables_initializer()
    sess.run(init_op)
    print(sess.run(tf.trainable_variables()))
    # 用ema.average(w1)获取w1的滑动平均值

    # 打印初始参数及滑动平均值
    print(sess.run([w1, ema.average(w1)]))

    # 参数w1的值赋为1
    sess.run(tf.assign(w1, 1))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 更新step和w1的值，模拟出100轮迭代后，参数w1变为10
    sess.run(tf.assign(global_step, 100))
    sess.run(tf.assign(w1, 10))
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    # 每次sess.run更新一次w1的滑动平均值
    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

    sess.run(ema_op)
    print(sess.run([w1, ema.average(w1)]))

