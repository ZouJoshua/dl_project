#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/16/19 1:30 PM
@File    : tf_optimizer.py
@Desc    : 优化器

"""


nb_epochs = 10
learning_rate = 0.01
data = None
loss_function = None

def evaluate_gradient(loss, data, params):
    pass

# 标准梯度下降

"""
先计算所有样本汇总误差，然后根据总误差来更新权值
for i in range(nb_epochs):
  params_grad = evaluate_gradient(loss_function, data, params)
  params = params - learning_rate * params_grad
缺点：
在一次更新中，就整个数据集计算梯度，计算非常慢
Batch gradient descent 对于凸函数可以收敛到全局极小值，对于非凸函数可以收敛到局部极小值。

"""


# 随机梯度下降

"""
随机抽取一个样本来计算误差，然后更新权值
for i in range(nb_epochs):
  np.random.shuffle(data)
  for example in data:
    params_grad = evaluate_gradient(loss_function, example, params)
    params = params - learning_rate * params_grad

缺点：
引入比较多的噪声，权值方向不一定正确

"""


# 批量梯度下降

"""
总样本中选取一个batch，然后计算这个batch的总误差，根据总误差来更新权值
for i in range(nb_epochs):
  np.random.shuffle(data)
  for batch in get_batches(data, batch_size=50):
    params_grad = evaluate_gradient(loss_function, batch, params)
    params = params - learning_rate * params_grad

优点：
MBGD 每一次利用一小批样本，可以降低参数更新时的方差，收敛更稳定，另一方面可以充分地利用深度学习库中高度优化的矩阵操作来进行更有效的梯度计算。


"""
