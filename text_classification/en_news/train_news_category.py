#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 2019/3/14 14:22
@File    : train_top_category.py
@Desc    : 训练英语新闻一级分类
"""

import os
from os.path import dirname
import sys
root_path = dirname(dirname(dirname(os.path.realpath(__file__))))
root_nlp_path = dirname(dirname(os.path.realpath(__file__)))
sys.path.append(root_path)
sys.path.append(root_nlp_path)


import json
import fasttext
from preprocess.preprocess_tools import CleanDoc
from evaluate import evaluate_model
from sklearn.model_selection import StratifiedKFold
import time
import random
import string
import re



class TopCategoryModel(object):

    def __init__(self, dataDir, category='top', k=5, model_level='one_level'):
        self._level = model_level
        if self._level == 'one_level':
            self.cg = 'top'
        else:
            self.cg = category
        self.k = k
        if os.path.exists(dataDir) and os.path.isdir(dataDir):
            self._datadir = dataDir
        else:
            raise Exception('数据路径不存在，请检查路径')
        self.label_idx_map = {"national": 4, "tech": 10,
                            "sports": 6, "science": 9,
                            "international": 3, "business": 8,
                            "entertainment": 15, "lifestyle": 12,
                            "auto": 11}
        self.idx_label_map = dict((str(k), v) for v, k in self.label_idx_map.items())

    def preprocess_data(self):
        print(">>>>> 正在预处理数据")
        fnames = os.listdir(self._datadir)
        datafiles = [os.path.join(self._datadir, fname) for fname in fnames if fname == 'train_corpus']
        data_all = list()
        class_cnt = dict()
        s = time.time()
        for datafile in datafiles:
            print(">>>>> 正在处理【{}】文件数据".format(datafile))
            dataf = open(datafile, 'r', encoding='utf-8')
            data = dataf.readlines()
            random.shuffle(data)
            # data_count = len(data)
            for li in data:
                line = li.strip('\n')
                dataX, dataY = self._preline(line)
                if dataY in class_cnt and dataX != "":
                    class_cnt[dataY] += 1
                elif dataX != "":
                    class_cnt[dataY] = 1
                if class_cnt[dataY] < 30001 and dataX != "":
                    data_all.append(line)
                else:
                    continue
            dataf.close()
        e = time.time()
        print('>>>>> 数据分类耗时：\n{}'.format(e - s))
        print('>>>>> 所有数据分类情况:\n{}'.format(json.dumps(class_cnt, indent=4)))
        self._generate_kfold_data(data_all)
        return

    def clean_title(self, text):
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = text.lower()
        no_emoji = CleanDoc(text)._remove_emoji(text)
        del_symbol = string.punctuation  # ASCII 标点符号，数字
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = no_emoji.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        text = re.sub(r"\s+", " ", text)
        return text


    def _preline(self, line):
        line_json = json.loads(line)
        title = line_json["title"]
        content = ""
        dataY = line_json[self._level]
        if "content" in line_json:
            content = line_json["content"]
        elif "html" in line_json:
            content = self._parse_html(line_json["html"])
        else:
            content = ""
        # dataX = CleanDoc(title + '.' + content).text  # 清洗数据
        dataX = self.clean_title(title)
        # _data = dataX + "\t__label__" + dataY
        return dataX, dataY

    def _generate_kfold_data(self, data_all):
        """
        分层k折交叉验证
        :param train_format_data:
        :return:
        """
        print(">>>>> 正在分割处理数据集")
        s = time.time()
        random.shuffle(data_all)
        datax = [self._preline(i)[0] for i in data_all]
        datay = [self._preline(i)[1] for i in data_all]
        e1 = time.time()
        print('>>>>> 数据分X\Y耗时{}'.format(e1 - s))
        skf = StratifiedKFold(n_splits=self.k)
        i = 0
        for train_index, test_index in skf.split(datax, datay):
            i += 1
            e2 = time.time()
            train_data = [datax[i] + "\t__label__" + str(self.label_idx_map[datay[i]]) for i in train_index]
            test_data = [datax[j] + "\t__label__" + str(self.label_idx_map[datay[j]]) for j in test_index]
            train_label_count = self._label_count([datay[i] for i in train_index])
            test_label_count = self._label_count([datay[j] for j in test_index])
            train_check = [data_all[i] for i in train_index]
            test_check = [data_all[i] for i in test_index]
            e3 = time.time()
            print('>>>>> 数据分训练集、测试集耗时{}'.format(e3 - e2))
            model_data_path = self._mkdir_path(i)
            train_file = os.path.join(model_data_path, 'train.txt')
            test_file = os.path.join(model_data_path, 'test.txt')
            train_check_file = os.path.join(model_data_path, 'train_check.json')
            test_check_file = os.path.join(model_data_path, 'test_check.json')
            self.write_file(train_file, train_data, 'txt')
            self.write_file(test_file, test_data, 'txt')
            self.write_file(train_check_file, train_check, 'json')
            self.write_file(test_check_file, test_check, 'json')
            print('>>>>> 文件:【{}】\n训练数据类别统计：{}'.format(train_file, json.dumps(train_label_count, indent=4)))
            print('>>>>> 文件:【{}】\n测试数据类别统计：{}'.format(test_file, json.dumps(test_label_count, indent=4)))
            if i == 1:
                break

    def _label_count(self, label_list):
        label_count = dict()
        for i in label_list:
            if i in label_count:
                label_count[i] += 1
            else:
                label_count[i] = 1
        return label_count

    def _mkdir_path(self, i):
        data_path = os.path.join(self._datadir, "{}_model_{}".format(self.cg, i))
        if not os.path.exists(data_path):
            # os.mkdir(data_path)
            model_data_path = os.path.join(data_path, "data")
            os.makedirs(model_data_path)
            return model_data_path
        else:
            raise Exception('已存在该路径')

    def write_file(self, file, data, file_format='txt'):
        print(">>>>> 正在写入文件【{}】".format(file))
        s = time.time()
        with open(file, 'w', encoding='utf-8') as f:
            if file_format == 'txt':
                for line in data:
                    f.write(line)
                    f.write('\n')
            elif file_format == 'json':
                for line in data:
                    # line_json = json.dumps(line)
                    f.write(line)
                    f.write('\n')
        e = time.time()
        print('<<<<<< 写文件耗时{}'.format(e -s))
        return

    def train_model(self):
        # self.preprocess_data()
        print(">>>>> 开始训练模型")
        train_precision = dict()
        test_precision = dict()
        for i in range(self.k):
            s = time.time()
            _model = "{}_model_{}".format(self.cg, i+1)
            data_path = os.path.join(self._datadir, _model)
            if os.path.exists(data_path):
                model_path = os.path.join(data_path, '{}_classification_model'.format(self.cg))
                train_data_path = os.path.join(data_path, 'data', 'train.txt')
                test_data_path = os.path.join(data_path, 'data', 'test.txt')
                test_check_path = os.path.join(data_path, 'data', 'test_check.json')
                test_check_pred_path = os.path.join(data_path, 'data', 'test_check_pred.json')
                train_check_path = os.path.join(data_path, 'data', 'train_check.json')
                train_check_pred_path = os.path.join(data_path, 'data', 'train_check_pred.json')
                classifier = fasttext.supervised(train_data_path, model_path, label_prefix="__label__", lr=0.1, epoch=20, dim=128, word_ngrams=3, loss='hs', bucket=20000)
                train_pred = classifier.test(train_data_path)
                test_pred = classifier.test(test_data_path)
                train_precision["model_{}".format(i+1)] = train_pred.precision
                test_precision["model_{}".format(i+1)] = test_pred.precision
                print("<<<<< 在训练集{}上的准确率：\n{}".format(_model, train_pred.precision))
                print("<<<<< 在测试集{}上的准确率：\n{}".format(_model, test_pred.precision))
                e = time.time()
                print('训练模型耗时{}'.format(e - s))
                self._predict(classifier, train_check_path, train_check_pred_path)
                self._predict(classifier, test_check_path, test_check_pred_path)
                self.evaluate_model(test_check_pred_path, self._level, _model)
            else:
                pass
        return train_precision, test_precision

    def _predict(self, classifier, json_file, json_out_file):
        with open(json_file, 'r', encoding='utf-8') as jfile, \
                open(json_out_file, 'w', encoding='utf-8') as joutfile:
            s = time.time()
            lines = jfile.readlines()
            _count = 0
            for line in lines:
                _line = json.loads(line)
                _data = self._preline(line)[0]
                _count += 1
                # if _count < 2:
                #     print(_data)
                labels = classifier.predict_proba([_data])
                _line['predict_{}'.format(self._level)] = self.idx_label_map[labels[0][0][0].replace("'", "").replace("__label__", "")]
                # print(line['predict_top_category'])
                _line['predict_{}_proba'.format(self._level)] = labels[0][0][1]
                joutfile.write(json.dumps(_line) + "\n")
                del line
                del _line
            e = time.time()
            print('<<<<< 预测及写入文件【{}】耗时{}'.format(json_out_file, (e - s)))
    def evaluate_model(self, datapath, model_level, model_num):
        return evaluate_model(datapath, model_level, model_num)

    def _get_label(self, jsonfile):
        with open(jsonfile, 'r', encoding='utf-8') as jfile:
            line = json.load(jfile)
        if self._level == 'one_level':
            label_idx_map = line[self._level]
        else:
            if line[self._level][self.cg]:
                label_idx_map = line[self._level][self.cg]
            else:
                raise Exception('请检查 label 文件')
        idx_label_map = dict()
        for key, value in label_idx_map.items():
            if value in idx_label_map:
                idx_label_map[value] = '{}+{}'.format(idx_label_map[value], key)
            else:
                idx_label_map[value] = key
        return label_idx_map, idx_label_map

    def _parse_html(self, html):
        # TODO:解析html内容
        pass

if __name__ == '__main__':
    s = time.time()
    dataDir = "/home/zoushuai/algoproject/tf_project/data/news_category"
    top_model = TopCategoryModel(dataDir, category='top', k=5, model_level='one_level')
    top_model.preprocess_data()
    train_precision, test_precision = top_model.train_model()
    e = time.time()
    print('训练一级分类模型耗时{}'.format(e - s))

