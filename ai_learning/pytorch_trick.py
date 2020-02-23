#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2/23/20 4:28 PM
@File    : pytorch_trick.py
@Desc    : pytorch trick

"""

import os

# 1.指定GPU编号
#设置当前使用的GPU设备仅为0号设备，设备名称为 /gpu:0：
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#设置当前使用的GPU设备为0,1号两个设备，名称依次为 /gpu:0、/gpu:1：
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 根据顺序表示优先使用0号设备,然后使用1号设备。


# 2.查看模型每层输出详情
"""
git clone https://github.com/sksq96/pytorch-summary
from torchsummary import summary
summary(your_model, input_size=(channels, H, W))  # input_size 是根据你自己的网络模型的输入尺寸进行设置。
"""

# 3.梯度裁剪

"""
import torch.nn as nn

outputs = model(data)
loss= loss_fn(outputs, target)
optimizer.zero_grad()
loss.backward()
nn.utils.clip_grad_norm_(model.parameters(), max_norm=20, norm_type=2)
optimizer.step()

nn.utils.clip_grad_norm_ 的参数：
  parameters – 一个基于变量的迭代器，会进行梯度归一化
  max_norm – 梯度的最大范数
  norm_type – 规定范数的类型，默认为L2
"""

# 梯度裁剪在某些任务上会额外消耗大量的计算时间


# 4.扩展单张图片维度
# 方法1

# tensor.unsqueeze(dim)：扩展维度，dim指定扩展哪个维度
# tensor.squeeze(dim)：去除dim指定的且size为1的维度，维度大于1时，squeeze()不起作用，不指定dim时，去除所有size为1的维度。
"""
import cv2
import torch

image = cv2.imread(img_path)
image = torch.tensor(image)
print(image.size())

img = image.unsqueeze(dim=0)  
print(img.size())

img = img.squeeze(dim=0)
print(img.size())

# output:
# torch.Size([(h, w, c)])
# torch.Size([1, h, w, c])
# torch.Size([h, w, c])
"""

# 方法2

"""
import cv2
import numpy as np

image = cv2.imread(img_path)
print(image.shape)
img = image[np.newaxis, :, :, :]
print(img.shape)

# output:
# (h, w, c)
# (1, h, w, c)
"""


# 方法3

"""
import cv2
import numpy as np

image = cv2.imread(img_path)
print(image.shape)
img = image[np.newaxis, :, :, :]
print(img.shape)

# output:
# (h, w, c)
# (1, h, w, c)
"""



# 5.one hot编码

# 在PyTorch中使用交叉熵损失函数的时候会自动把label转化成onehot，所以不用手动转化，
# 而使用MSE需要手动转化成onehot编码。

"""
import torch
class_num = 8
batch_size = 4

def one_hot(label):
    #将一维列表转换为独热编码
    label = label.resize_(batch_size, 1)
    m_zeros = torch.zeros(batch_size, class_num)
    # 从 value 中取值，然后根据 dim 和 index 给相应位置赋值
    onehot = m_zeros.scatter_(1, label, 1)  # (dim,index,value)

    return onehot.numpy()  # Tensor -> Numpy

label = torch.LongTensor(batch_size).random_() % class_num  # 对随机数取余
print(one_hot(label))

# output:
[[0. 0. 0. 1. 0. 0. 0. 0.]
 [0. 0. 0. 0. 1. 0. 0. 0.]
 [0. 0. 1. 0. 0. 0. 0. 0.]
 [0. 1. 0. 0. 0. 0. 0. 0.]]

"""

# 6.防止验证模型时爆显存
# 验证模型时不需要求导，即不需要梯度计算，关闭autograd，可以提高速度，节约内存。如果不关闭可能会爆显存。
"""
with torch.no_grad():
    # 使用model进行预测的代码
    pass
"""

# PyTorch的缓存分配器会事先分配一些固定的显存，即使实际上tensors并没有使用完这些显存，
# 这些显存也不能被其他应用使用。这个分配过程由第一次CUDA内存访问触发的。
# 而 torch.cuda.empty_cache() 的作用就是释放缓存分配器当前持有的且未占用的缓存显存，
# 以便这些显存可以被其他GPU应用程序中使用，并且通过 nvidia-smi命令可见。
# 注意使用此命令不会释放tensors占用的显存。

# 7.学习率衰减

"""
import torch.optim as optim
from torch.optim import lr_scheduler

# 训练前的初始化
optimizer = optim.Adam(net.parameters(), lr=0.001)
scheduler = lr_scheduler.StepLR(optimizer, 10, 0.1)  # # 每过10个epoch，学习率乘以0.1

# 训练过程中
for n in n_epoch:
    scheduler.step()
    ...
"""

# 8.冻结某些层的参数

# 需要先知道每一层的名字，通过如下代码打印
"""
net = Network()  # 获取自定义网络结构
for name, value in net.named_parameters():
    print('name: {0},\t grad: {1}'.format(name, value.requires_grad))
    
name: cnn.VGG_16.convolution1_1.weight,	 grad: True
name: cnn.VGG_16.convolution1_1.bias,	 grad: True
name: cnn.VGG_16.convolution1_2.weight,	 grad: True
name: cnn.VGG_16.convolution1_2.bias,	 grad: True
name: cnn.VGG_16.convolution2_1.weight,	 grad: True
"""

# 定义一个要冻结的层的列表：

"""
no_grad = [
    'cnn.VGG_16.convolution1_1.weight',
    'cnn.VGG_16.convolution1_1.bias',
    'cnn.VGG_16.convolution1_2.weight',
    'cnn.VGG_16.convolution1_2.bias'
]
"""

# 冻结方法如下
"""
net = Net.CTPN()  # 获取网络结构
for name, value in net.named_parameters():
    if name in no_grad:
        value.requires_grad = False
    else:
        value.requires_grad = True
在定义优化器时，只对requires_grad为True的层的参数进行更新。

optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=0.01)

"""

# 9.对不同层使用不同学习率


"""
net = Network()  # 获取自定义网络结构
for name, value in net.named_parameters():
    print('name: {}'.format(name))

# 输出：
# name: cnn.VGG_16.convolution1_1.weight
# name: cnn.VGG_16.convolution1_1.bias
# name: cnn.VGG_16.convolution1_2.weight
# name: cnn.VGG_16.convolution1_2.bias
# name: cnn.VGG_16.convolution2_1.weight
"""

# 对 convolution1 和 convolution2 设置不同的学习率，首先将它们分开，即放到不同的列表里：
# 我们将模型划分为两部分，存放到一个列表里，每部分就对应上面的一个字典，在字典里设置不同的学习率。
# 当这两部分有相同的其他参数时，就将该参数放到列表外面作为全局参数，如上面的`weight_decay`。
# 也可以在列表外设置一个全局学习率，当各部分字典里设置了局部学习率时，就使用该学习率，否则就使用列表外的全局学习率。

"""
conv1_params = []
conv2_params = []

for name, parms in net.named_parameters():
    if "convolution1" in name:
        conv1_params += [parms]
    else:
        conv2_params += [parms]

# 然后在优化器中进行如下操作：
optimizer = optim.Adam(
    [
        {"params": conv1_params, 'lr': 0.01},
        {"params": conv2_params, 'lr': 0.001},
    ],
    weight_decay=1e-3,
)
"""

# 10.模型相关操作
# 11.pytorch内置的one hot函数

# Pytorch 1.1后，one_hot可以直接用
"""
import torch.nn.functional as F
import torch

tensor =  torch.arange(0, 5) % 3  # tensor([0, 1, 2, 0, 1])
one_hot = F.one_hot(tensor)

# 输出：
# tensor([[1, 0, 0],
#         [0, 1, 0],
#         [0, 0, 1],
#         [1, 0, 0],
#         [0, 1, 0]])

F.one_hot会自己检测不同类别个数，生成对应独热编码。我们也可以自己指定类别数：


tensor =  torch.arange(0, 5) % 3  # tensor([0, 1, 2, 0, 1])
one_hot = F.one_hot(tensor, num_classes=5)
# 输出：
# tensor([[1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0],
#         [0, 0, 1, 0, 0],
#         [1, 0, 0, 0, 0],
#         [0, 1, 0, 0, 0]])

"""


# 12.网络参数初始化

# 方法1
"""
https://ptorch.com/docs/1/nn-init
常用的初始化操作，例如正态分布、均匀分布、xavier初始化、kaiming初始化等都已经实现，可以直接使用

init.xavier_uniform(net1[0].weight)

"""


# 方法2
"""
对于自定义的初始化方法，有时tensor的功能不如numpy强大灵活，故可以借助numpy实现初始化方法，再转换到tensor上使用。

for layer in net1.modules():
    if isinstance(layer, nn.Linear): # 判断是否是线性层
        param_shape = layer.weight.shape
        layer.weight.data = torch.from_numpy(np.random.normal(0, 0.5, size=param_shape)) 
        # 定义为均值为 0，方差为 0.5 的正态分布
"""

# 13.加载内置预训练模型


# torchvision.models模块的子模块中包含以下模型：
#
# AlexNet
# VGG
# ResNet
# SqueezeNet
# DenseNet

"""
import torchvision.models as models
resnet18 = models.resnet18()
alexnet = models.alexnet()
vgg16 = models.vgg16()

"""

# 有一个很重要的参数为pretrained，默认为False，表示只导入模型的结构，其中的权重是随机初始化的。
# 如果pretrained 为 True，表示导入的是在ImageNet数据集上预训练的模型。

"""
import torchvision.models as models
resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
"""