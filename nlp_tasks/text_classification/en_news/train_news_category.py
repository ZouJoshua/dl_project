#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2019/3/14 14:22
@File    : train_top_category.py
@Desc    : 训练英语新闻一级分类
"""

import os
import json
import time
import random
import re
import string
from sklearn.model_selection import StratifiedKFold

from preprocess.preprocess_tools import CleanDoc, read_json_format_file, write_file
from model_normal.fasttext_model import FastTextClassifier
from evaluate.eval_calculate import EvaluateModel


import logging
from utils.logger import Logger
from setting import LOG_PATH

log_file = os.path.join(LOG_PATH, 'fasttext_train_log')
log = Logger("fasttext_train_log", log2console=True, log2file=True, logfile=log_file).get_logger()

class DataSet(object):

    def __init__(self, data_path, business_type='news_category', k=5, logger=None):
        if os.path.exists(data_path) and os.path.isdir(data_path):
            self.data_path = data_path
        else:
            raise Exception('数据路径不存在，请检查路径')

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("fasttext_train_log")
            self.log.setLevel(logging.INFO)
        self.k = k
        self.bt = business_type
        self.label_idx_map = {"national": 4, "tech": 10,
                            "sports": 6, "science": 9,
                            "international": 3, "business": 8,
                            "entertainment": 15, "lifestyle": 12,
                            "auto": 11}
        self.idx_label_map = dict((str(k), v) for v, k in self.label_idx_map.items())
        self.split_dataset()

    def split_dataset(self):
        self.log.info("预处理数据文件...")
        # print(">>>>> 预处理数据文件...")
        fnames = os.listdir(self.data_path)
        datafiles = [os.path.join(self.data_path, fname) for fname in fnames]
        data_all = list()
        class_cnt = dict()
        s = time.time()
        for datafile in datafiles:
            # print(">>>>> 正在处理数据文件：{}".format(datafile))
            self.log.info("正在处理数据文件:{}".format(datafile))
            for line in read_json_format_file(datafile):
                # dataY = line["one_level"]
                if self._preline(line):
                    dataX, dataY = self._preline(line).split('\t__label__')
                    dataY = line["one_level"]
                    if str(dataY) in class_cnt:
                        class_cnt[str(dataY)] += 1
                    else:
                        class_cnt[str(dataY)] = 1
                    if class_cnt[str(dataY)] < 40001 and dataX != "":
                        data_all.append(line)
                    else:
                        continue
        e = time.time()
        self.log.info('数据分类耗时： {}s'.format(e - s))
        self.log.info('所有数据分类情况： {}'.format(json.dumps(class_cnt, indent=4)))
        self._generate_kfold_data(data_all)
        return

    def _generate_kfold_data(self, data_all):
        """
        按照label分层数据
        :param train_format_data:
        :return:
        """
        s = time.time()
        random.shuffle(data_all)
        datax = [self._preline(i).split('\t__label__')[0] for i in data_all]
        datay = [self._preline(i).split('\t__label__')[1] for i in data_all]
        e1 = time.time()
        self.log.info('数据分X\Y耗时： {}s'.format(e1 - s))

        skf = StratifiedKFold(n_splits=self.k)
        i = 0
        for train_index, test_index in skf.split(datax, datay):
            i += 1
            e2 = time.time()
            train_label_count = self._label_count([datay[i] for i in train_index])
            test_label_count = self._label_count([datay[j] for j in test_index])
            train_data = [self._preline(data_all[i]) for i in train_index]
            test_data = [self._preline(data_all[j]) for j in test_index]
            train_check = [data_all[i] for i in train_index]
            test_check = [data_all[i] for i in test_index]
            e3 = time.time()
            self.log.info('数据分训练集、测试集耗时： {}s'.format(e3 - e2))

            model_data_path = self._mkdir_path(i)
            train_file = os.path.join(model_data_path, 'train.txt')
            test_file = os.path.join(model_data_path, 'test.txt')
            train_check_file = os.path.join(model_data_path, 'train_check.json')
            test_check_file = os.path.join(model_data_path, 'test_check.json')
            write_file(train_file, train_data, 'txt')
            write_file(test_file, test_data, 'txt')
            write_file(train_check_file, train_check, 'json')
            write_file(test_check_file, test_check, 'json')

            self.log.info('文件:{}\n训练数据类别统计：{}'.format(train_file, json.dumps(train_label_count, indent=4)))
            self.log.info('文件:{}\n测试数据类别统计：{}'.format(test_file, json.dumps(test_label_count, indent=4)))
            if i == 1:
                break

    def _preline(self, line_json):
        if not isinstance(line_json, dict):
            self.log.error("该文本行不是json类型")
            raise Exception("该文本行不是json类型")
        title = line_json["title"]
        content = ""
        dataY = str(self.label_idx_map.get(line_json["one_level"]))

        if "content" in line_json:
            content = line_json["content"]
        elif "html" in line_json:
            content = self._parse_html(line_json["html"])
        # dataX = clean_string((title + '.' + content).lower())  # 清洗数据
        # dataX = CleanDoc(title.lower()).text  # 清洗数据
        dataX = self.clean_title(title)  # 清洗数据

        if dataX:
            _data = dataX + "\t__label__" + dataY
            return _data
        else:
            return None


    def _label_count(self, label_list):
        label_count = dict()
        for i in label_list:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count

    def _mkdir_path(self, i):
        curr_data_path = os.path.join(self.data_path, "{}_model_{}".format(self.bt, i))
        if not os.path.exists(curr_data_path):
            # os.mkdir(data_path)
            model_data_path = os.path.join(curr_data_path, "data")
            os.makedirs(model_data_path)
            return model_data_path
        else:
            raise Exception('已存在该路径')


    def clean_title(self, text):
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = text.lower()
        no_emoji = CleanDoc(text).remove_emoji(text)
        del_symbol = string.punctuation  # ASCII 标点符号
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = no_emoji.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        text = re.sub(r"\s+", " ", text)
        return text


    def _parse_html(self,html):
        pass

class NewsCategoryModel(object):

    def __init__(self, data_path, business_type='news_category', k=5, logger=None):
        if os.path.exists(data_path) and os.path.isdir(data_path):
            self.data_path = data_path
        else:
            raise Exception('数据路径不存在，请检查路径')

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("fasttext_train_log")
            self.log.setLevel(logging.INFO)
        self.k = k
        self.bt = business_type
        self.label_idx_map = {"national": 4, "tech": 10,
                            "sports": 6, "science": 9,
                            "international": 3, "business": 8,
                            "entertainment": 15, "lifestyle": 12,
                            "auto": 11}
        self.idx_label_map = dict((str(k), v) for v, k in self.label_idx_map.items())

    def train_model(self):
        for i in range(self.k):
            s = time.time()
            _model = "{}_model_{}".format(self.bt, i+1)
            _data_path = os.path.join(self.data_path, _model)
            if os.path.exists(_data_path):
                model_path = os.path.join(_data_path, '{}_model'.format(self.bt))
                train_test_data_path = os.path.join(_data_path, 'data')
                classifier = FastTextClassifier(model_path, train=True, file_path=train_test_data_path, logger=self.log)
                test_check_path = os.path.join(train_test_data_path, 'test_check.json')
                test_check_pred_path = os.path.join(train_test_data_path, 'test_check_pred.json')
                train_check_path = os.path.join(train_test_data_path, 'train_check.json')
                train_check_pred_path = os.path.join(train_test_data_path, 'train_check_pred.json')
                e = time.time()
                self.log.info('训练模型耗时： {}s'.format(e - s))
                # self.predict2file(classifier, train_check_path, train_check_pred_path)
                # self.predict2file(classifier, test_check_path, test_check_pred_path)
                label_list = sorted([self.idx_label_map.get(i.replace("__label__", "")) for i in classifier.model.labels])
                self.log.info("模型标签：\n{}".format(label_list))
                # self.evaluate_model(test_check_pred_path, "one_level", labels=label_list)
            else:
                continue
        return

    def predict2file(self, classifier, json_file, json_out_file):
        with open(json_out_file, 'w', encoding='utf-8') as joutfile:
            s = time.time()
            for line in read_json_format_file(json_file):
                _data = self._preline(line)
                if _data:
                    labels = classifier.predict([_data])
                    line['predict_one_level'] = self.idx_label_map[labels[0][0][0].replace("'", "").replace("__label__", "")]
                    # print(line['predict_top_category'])
                    line['predict_one_level_proba'] = labels[0][0][1]
                    joutfile.write(json.dumps(line) + "\n")
                    del line
                else:
                    continue
            e = time.time()
            self.log.info('预测及写入文件耗时： {}s'.format(e - s))

    def _preline(self, line_json):
        if not isinstance(line_json, dict):
            self.log.error("该文本行不是json类型")
            raise Exception("该文本行不是json类型")
        title = line_json["title"]
        # dataX = clean_string((title + '.' + content).lower())  # 清洗数据
        dataX = self.clean_title(title)  # 清洗数据
        if dataX:
            _data = dataX
            return _data
        else:
            return None

    def clean_title(self, text):
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = text.lower()
        no_emoji = CleanDoc(text).remove_emoji(text)
        del_symbol = string.punctuation  # ASCII 标点符号，数字
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = no_emoji.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        text = re.sub(r"\s+", " ", text)
        return text

    def evaluate_model(self, datapath, key_, labels=None):
        em = EvaluateModel(datapath, key_name=key_, logger=self.log, label_names=labels)
        return em.evaluate_model_v2()


if __name__ == '__main__':
    s = time.time()
    dataDir = "/data/en_news"
    # dataDir = "/data/emotion_analysis/taste_ft_model"
    # DataSet(dataDir, logger=log)
    bcm = NewsCategoryModel(dataDir, logger=log)
    bcm.train_model()
    e = time.time()
    print('训练新闻title分类模型耗时{}'.format(e - s))

