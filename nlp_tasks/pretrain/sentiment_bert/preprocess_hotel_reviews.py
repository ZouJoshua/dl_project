#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2/22/20 10:06 PM
@File    : preprocess_hotel_reviews.py
@Desc    : 训练数据预处理

"""

import numpy as np
from sklearn.utils import shuffle
import os
import matplotlib.pyplot as plt

from setting import DATA_PATH




def get_corpus(hotel_reviews_dir):
    # all_positive和all_negative含有所有的正样本和负样本
    with open(hotel_reviews_dir + "/" + "neg.txt", "r", encoding="utf-8") as f:
        all_negative = [line.strip() for line in f.readlines()]
    with open(hotel_reviews_dir + "/" + "pos.txt", "r", encoding="utf-8") as f:
        all_positive = [line.strip() for line in f.readlines()]
    return all_negative, all_positive

def plot_text_len(all_negative, all_positive):
    # 获取所有文本的长度
    all_length = [len(i) for i in all_negative] + [len(i) for i in all_positive]

    # 可视化语料序列长度, 可见大部分文本的长度都在300以下
    prop = np.mean(np.array(all_length) < 300)
    print("评论长度在300以下的比例: {}".format(prop))

    plt.hist(all_length, bins=30)
    plt.show()

def split_data_to_file(all_positive, all_negative, hotel_reviews_dir):
    """
    把所有的语料放到list里, 每一条语料是一个dict: {"text":文本, "label":分类}
    :param all_positive:
    :param all_negative:
    :return:
    """
    all_data = []
    for text in all_positive:
        all_data.append({"text": text, "label": 1})
    for text in all_negative:
        all_data.append({"text": text, "label": 0})


    # shuffle打乱顺序
    all_data = shuffle(all_data, random_state=1)

    # 拿出5%的数据用来测试
    test_proportion = 0.05
    test_idx = int(len(all_data) * test_proportion)


    # 分割训练集和测试集
    test_data = all_data[:test_idx]
    train_data = all_data[test_idx:]

    # 输出训练集和测试集为txt文件, 每一行为一个dict: {"text":文本, "label":分类}
    train_file = os.path.join(hotel_reviews_dir, "train_sentiment.txt")
    test_file = os.path.join(hotel_reviews_dir, "test_sentiment.txt")
    with open(train_file, "a", encoding="utf-8") as f:
        for line in train_data:
            f.write(str(line))
            f.write("\n")
    with open(test_file, "a", encoding="utf-8") as f:
        for line in test_data:
            f.write(str(line))
            f.write("\n")


def main():
    # 评论语料目录
    hotel_reviews_dir = os.path.join(DATA_PATH, "corpus", "hotel_reviews")
    all_neg, all_pos = get_corpus(hotel_reviews_dir)
    # plot_text_len(all_neg, all_pos)
    split_data_to_file(all_neg, all_pos, hotel_reviews_dir)

if __name__ == "__main__":
    main()