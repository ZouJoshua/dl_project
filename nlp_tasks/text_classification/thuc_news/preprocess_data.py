#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2/26/20 11:19 PM
@File    : preprocess_data.py
@Desc    : 

"""


import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import json
import tqdm
import random
from setting import DATA_PATH



def merge_files(data_dir, outfile):
    """
    合并原始文件到一个文件
    每行一个格式为{"text":"","label":""}
    :param dir:解压后的文件目录路径
    :return:
    """
    print("合并所有文件文本到一个文件")
    label_stat = dict()
    label_list = os.listdir(data_dir)
    data_dirs = [os.path.join(data_dir, label) for label in label_list]
    with open(outfile, "w", encoding="utf-8") as wf:
        for data_dir in data_dirs:
            file_names = os.listdir(data_dir)
            label = os.path.split(data_dir)[-1]
            print("正在处理>>>{}目录的文件".format(label))
            label_count = len(file_names)
            label_stat[label] = label_count
            all_files = [os.path.join(data_dir, file) for file in file_names]
            file_count = 0
            for file in all_files:
            # for file in tqdm.tqdm(all_files, desc="Reading Files of {}".format(label)):
                line = dict()
                line["label"] = label
                file_count += 1
                with open(file, "r", encoding="utf-8") as f:
                    text = f.read()
                    line["text"] = text
                    str_line = json.dumps(line, ensure_ascii=False)
                    wf.write(str_line + "\n")
                    wf.flush()
                if file_count % 10000 == 0:
                    print("已读取{}个文件".format(file_count))

        print("All files Down!!")
        print("各label的样本数量:\n{}".format(json.dumps(label_stat, ensure_ascii=False, indent=4)))


def split_dataset(file):
    """
    划分数据集为训练集\验证集\测试集,按照比例7:2:1,每个label 5w的样本
    :param file:
    :return:
    """
    print("划分数据集为:-训练集-验证集-测试集")
    print("正在处理的当前文件:{}".format(file))
    corpus_dir = os.path.split(file)[0]
    train_file = os.path.join(corpus_dir, "thuc_news.train.txt")
    eval_file = os.path.join(corpus_dir, "thuc_news.eval.txt")
    test_file = os.path.join(corpus_dir, "thuc_news.test.txt")
    with open(file, "r", encoding='utf-8') as f, \
            open(train_file, "w", encoding="utf-8") as train_f, \
            open(eval_file, "w", encoding="utf-8") as eval_f, \
            open(test_file, "w", encoding="utf-8") as test_f:
        lines = f.readlines()
        category_dict = dict()
        for line in lines:
            _line = json.loads(line.strip())
            label = _line["label"]
            if label in category_dict:
                category_dict[label].append(line)
            else:
                category_dict[label] = []
                category_dict[label].append(line)
        for label in category_dict.keys():
            lines = category_dict[label]
            random.shuffle(lines)
            lines_sample = lines[:50000]
            random.shuffle(lines_sample)
            nums = len(lines_sample)
            train_lines = lines_sample[:int(nums*0.7)]
            for line in tqdm.tqdm(train_lines, desc="Writing train data of {}".format(label)):
                train_f.write(line)

            eval_lines = lines_sample[int(nums*0.7):int(nums*0.9)]
            for line in tqdm.tqdm(eval_lines, desc="Writing eval data of {}".format(label)):
                eval_f.write(line)
            test_lines = lines_sample[int(nums*0.9):]
            for line in tqdm.tqdm(test_lines, desc="Writing test data of {}".format(label)):
                test_f.write(line)



def plot_text_len(file):
    """
    文本长度可视化
    :param file:
    :return:
    """
    with open(file, "r", encoding='utf-8') as f:
        lines = f.readlines()
    # 获取所有文本的长度
    all_length = [len(i.strip().split("__label__")[0].split(" ")) for i in lines]
    # error_text = [i.strip().split("__label__")[0] for i in lines]
    for i in lines:
        a = i.strip().split("__label__")[0]
        if len(a) < 5:
            print(i)

    print(all_length[:2])
    # 可视化语料序列长度, 可见大部分文本的长度都在300以下
    prop = np.mean(np.array(all_length) < 1000)
    print("评论长度在1000以下的比例: {}".format(prop))

    plt.hist(all_length, bins=500)
    plt.show()





def main():
    origin_dir = "/data/common/thucnews_data"
    corpus_dir = os.path.join(DATA_PATH, "corpus", "thuc_news")
    corpus_file = os.path.join(corpus_dir, "thuc_news.all.txt")
    train_file = os.path.join(corpus_dir, "train.txt")

    # 合并文件
    # merge_files(origin_dir, corpus_file)
    # 划分数据集
    # split_dataset(corpus_file)
    # 所有文本长度
    plot_text_len(train_file)


if __name__ == '__main__':
    main()