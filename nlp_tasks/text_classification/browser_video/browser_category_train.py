#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-12 下午3:06
@File    : browser_category_train.py
@Desc    : 训练浏览器视频分类
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

from nltk.corpus import stopwords


import logging
from utils.logger import Logger
from setting import LOG_PATH, CONFIG_PATH
import fasttext


log_file = os.path.join(LOG_PATH, 'fasttext_train_log')
log = Logger("fasttext_train_log", log2console=True, log2file=True, logfile=log_file).get_logger()

class DataSet(object):

    def __init__(self, data_path, feature_model, business_type='browser_category', k=5, logger=None):
        if os.path.exists(data_path) and os.path.isdir(data_path):
            self.data_path = data_path
        else:
            raise Exception('数据路径不存在，请检查路径')

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("fasttext_train_log")
            self.log.setLevel(logging.INFO)
        self.fe = FeatureExtract(feature_model, logger=self.log)
        self.k = k
        self.bt = business_type
        self.split_dataset()

    def split_dataset(self):
        self.log.info("预处理数据文件...")
        # print(">>>>> 预处理数据文件...")
        fnames = os.listdir(self.data_path)
        self.pre_file = os.path.join(self.data_path, "raw_data_clean")
        datafiles = [os.path.join(self.data_path, fname) for fname in fnames]
        class_cnt = dict()
        s = time.time()
        if not os.path.exists(self.pre_file):
            with open(self.pre_file, 'w') as f:
                for datafile in datafiles:
                    # print(">>>>> 正在处理数据文件：{}".format(datafile))
                    if os.path.isfile(datafile):
                        self.log.info("正在处理数据文件:{}".format(datafile))
                        for line in read_json_format_file(datafile):
                            label = str(line["category"])
                            if label in class_cnt:
                                class_cnt[label] += 1
                            else:
                                class_cnt[label] = 1
                            line["pre_clean_text"] = self._pre_clean_line(line)
                            f.write(json.dumps(line) + "\n")
                    else:
                        self.log.error("处理数据文件时遇到目录：{}".format(datafile))
            e = time.time()
            self.log.info('数据分类耗时： {}s'.format(e - s))
            self.log.info('所有数据分类情况： {}'.format(json.dumps(class_cnt, indent=4)))

        pre_data_all, ori_data_all = self.pre_train_file()
        self._generate_kfold_data(pre_data_all, ori_data_all)
        return

    def pre_train_file(self):
        self.log.info('准备模型训练文件...')
        class_cnt = dict()
        ori_data_all = list()
        pre_data_all = list()
        for line in read_json_format_file(self.pre_file):
            label = str(line["category"])
            text = line["pre_clean_text"]
            if label in ['211', '212', '213', '214', '215', '216', '217', '218', '219', '220', '221', '222',
                         '223', '224', '225', '226', '227', '228', '229', '230']:
                if text != "":
                    if label in class_cnt:
                        class_cnt[label] += 1
                    else:
                        class_cnt[label] = 1

                    if class_cnt[label] < 30000:
                        if label in ["211", "212", "213", "214", "226", "229", "230", "222", "216", "227", "223"]:
                            dataY_tmp = "200"
                        else:
                            dataY_tmp = label
                        pre_data_all.append((text, label, dataY_tmp))
                        ori_data_all.append(line)
                    else:
                        continue
            else:
                self.log.warning("分类不在211-230内")
        return pre_data_all, ori_data_all

    def _generate_kfold_data(self, data_all, ori_data_all):
        """
        按照label分层数据
        :param train_format_data:
        :return:
        """
        s = time.time()
        random.shuffle(data_all)
        datax = [i[0] for i in data_all]
        datay = [i[1] for i in data_all]
        e1 = time.time()
        self.log.info('数据分X\Y耗时： {}s'.format(e1 - s))

        skf = StratifiedKFold(n_splits=self.k)
        i = 0
        for train_index, test_index in skf.split(datax, datay):
            i += 1
            e2 = time.time()
            train_label_count = self._label_count([datay[i] for i in train_index])
            test_label_count = self._label_count([datay[j] for j in test_index])
            train_data = ["{}\t__label__{}".format(data_all[i][0], data_all[i][2]) for i in train_index]
            test_data = ["{}\t__label__{}".format(data_all[j][0], data_all[j][2]) for j in test_index]
            train_check = [ori_data_all[i] for i in train_index]
            test_check = [ori_data_all[i] for i in test_index]
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

    def _pre_clean_line(self, line_json):
        if not isinstance(line_json, dict):
            self.log.error("该文本行不是json类型")
            raise Exception("该文本行不是json类型")
        title = line_json["article_title"]
        content = line_json["text"]
        tag_list = ",".join(line_json["name"])
        clean_x = self.clean_content(title + ' ' + content + " " + tag_list)
        return clean_x

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
        model_data_path = os.path.join(curr_data_path, "data")
        if not os.path.exists(curr_data_path):
            # os.mkdir(data_path)
            os.makedirs(model_data_path)
            return model_data_path
        else:
            self.log.warning('已存在该路径: {}'.format(model_data_path))
            return model_data_path


    def clean_title(self, text):
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = text.lower()
        no_emoji = CleanDoc(text).remove_emoji(text)
        del_symbol = string.punctuation  # ASCII 标点符号，数字
        remove_punctuation_map = dict((ord(char), " ") for char in del_symbol)
        text = no_emoji.translate(remove_punctuation_map)  # 去掉ASCII 标点符号
        text = re.sub(r"\s+", " ", text)
        return text

    def clean_content(self, text):
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = text.lower()
        cd_instance = CleanDoc(text)
        no_emoji = cd_instance.remove_emoji(text)
        no_url = cd_instance.clean_html(no_emoji)
        no_mail = cd_instance.clean_mail(no_url)
        no_symbol = cd_instance.remove_symbol(no_mail)
        text = re.sub(r"\s+", " ", no_symbol)
        return text


    def _parse_html(self,html):
        pass

class FeatureExtract(object):

    def __init__(self, feature_model, logger=None):
        self.model_path = feature_model

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("fasttext_train_log")
            self.log.setLevel(logging.INFO)
        self.model = self.load_model()

    def load_model(self):
        config_file = os.path.join(CONFIG_PATH, "fasttext_train.conf")
        classifier = FastTextClassifier(self.model_path, config_file, "fasttext.args", train=False, logger=self.log)
        return classifier

    def predict_feature(self, text):
        token_words = self.pre_text(text)
        # print(token_words)
        if not token_words.strip():
            feature_words = ""
        else:
            # print(token_words)
            result = self.model.predict(token_words, k=3)
            pred_prob = result[0][0][1] + result[0][1][1]
            if result[0][0][1] > 0.9:
                feature_words = result[0][0][0].replace('__label__', '')
            else:
                if pred_prob > 0.9:
                    feature_words = result[0][0][0].replace('__label__', '') + " " + result[0][1][0].replace('__label__',
                                                                                                             '')
                else:
                    feature_words = result[0][0][0].replace('__label__', '') + " " + result[0][1][0].replace('__label__',
                                                                                                             '') \
                                    + " " + result[0][2][0].replace('__label__', '')
        return feature_words


    def pre_text(self, text):

        l_text = text.lower()
        text_tok = l_text.split(' ')
        new_tokens_list = list()

        for tok in text_tok:
            tok = tok.replace("\r", " ").replace("\n", " ").replace("\t", " ")
            new_tok = CleanDoc(tok).remove_symbol(tok)
            new_tok = CleanDoc(new_tok).remove_emoji(new_tok)
            if new_tok and not new_tok.isdigit() and new_tok not in stopwords.words("english"):
                new_tokens_list.append(new_tok)
        token_words = " ".join(new_tokens_list)

        return token_words







class BrowserCategoryModel(object):

    def __init__(self, data_path, feature_model, business_type='browser_category', k=5, logger=None):
        if os.path.exists(data_path) and os.path.isdir(data_path):
            self.data_path = data_path
        else:
            raise Exception('数据路径不存在，请检查路径')

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("fasttext_train_log")
            self.log.setLevel(logging.INFO)
        self.fe = FeatureExtract(feature_model, logger=self.log)
        self.k = k
        self.bt = business_type

    def train_model(self):
        for i in range(self.k):
            s = time.time()
            _model = "{}_model_{}".format(self.bt, i+1)
            _data_path = os.path.join(self.data_path, _model)
            if os.path.exists(_data_path):
                model_path = os.path.join(_data_path, '{}_model'.format(self.bt))
                train_test_data_path = os.path.join(_data_path, 'data')
                config_file = os.path.join(CONFIG_PATH, "fasttext_train.conf")
                classifier = FastTextClassifier(model_path, config_file, "fasttext.args", train=False, file_path=train_test_data_path, logger=log)
                test_check_path = os.path.join(train_test_data_path, 'test_check.json')
                test_check_pred_path = os.path.join(train_test_data_path, 'test_check_pred.json')
                train_check_path = os.path.join(train_test_data_path, 'train_check.json')
                train_check_pred_path = os.path.join(train_test_data_path, 'train_check_pred.json')
                # sub_classifier = self.train_sub_model(train_check_path, test_check_path)
                sub_classifier = fasttext.load_model("/data/browser_category/train_v6/browser_category_model_1/data/browser_category_sub_model.bin")

                e = time.time()
                self.log.info('训练模型耗时： {}s'.format(e - s))
                self.predict2file(classifier, sub_classifier, train_check_path, train_check_pred_path)
                self.predict2file(classifier, sub_classifier, test_check_path, test_check_pred_path)
                main_model_labels = [i.replace("__label__", "") for i in classifier.model.labels if i.replace("__label__", "") != "200"]
                sub_model_labels = [i.replace("__label__", "") for i in sub_classifier.labels]
                label_list = sorted(main_model_labels + sub_model_labels)
                self.log.info("模型标签：\n{}".format(label_list))
                self.evaluate_model(test_check_pred_path, "category", labels=label_list)
            else:
                continue
        return
    def train_sub_model(self, train_check_path, test_check_path):

        _model = "{}_model_{}".format(self.bt, "1")
        _data_path = os.path.join(self.data_path, _model, 'data')

        sub_train_file = os.path.join(_data_path, 'sub_train.txt')
        sub_test_file = os.path.join(_data_path, 'sub_test.txt')

        if not os.path.exists(sub_train_file) and not os.path.exists(sub_test_file):
            with open(sub_train_file, 'w') as f:
                for line in read_json_format_file(train_check_path):
                    label = str(line["category"])
                    if label in ["211", "212", "213", "214", "226", "229", "230", "222", "216", "227", "223"]:
                        # _dataX = self._preline(line)
                        _dataX = line["pre_clean_text"]
                        new_line = "{}\t__label__{}\n".format(_dataX, label)
                        f.write(new_line)

            with open(sub_test_file, 'w') as f:
                for line in read_json_format_file(test_check_path):
                    label = str(line["category"])
                    if label in ["211", "212", "213", "214", "226", "229", "230", "222", "216", "227", "223"]:
                        # _dataX = self._preline(line)
                        _dataX = line["pre_clean_text"]
                        new_line = "{}\t__label__{}\n".format(_dataX, label)
                        f.write(new_line)
        model_path = os.path.join(_data_path, '{}_sub_model'.format(self.bt))

        s = time.time()
        model = fasttext.supervised(sub_train_file,
                                    model_path,
                                    label_prefix="__label__",
                                    epoch=20,
                                    dim=256,
                                    silent=False,
                                    lr=0.1,
                                    minn=2,
                                    maxn=4,
                                    loss='ns',
                                    min_count=1,
                                    word_ngrams=4,
                                    bucket=2000)
        train_result = model.test(sub_train_file)
        e = time.time()
        self.log.info('训练次级模型耗时： {}s'.format(e - s))
        self.log.info('次级模型训练集准确率： {}'.format(train_result.precision))
        test_result = model.test(sub_test_file)
        self.log.info('次级模型测试集准确率: {}'.format(test_result.precision))
        return model


    def predict2file(self, classifier, sub_classifier, json_file, json_out_file):
        with open(json_out_file, 'w', encoding='utf-8') as joutfile:
            s = time.time()
            for line in read_json_format_file(json_file):
                # label = str(line["category"])
                # if label in ["211", "212", "213", "214", "226", "229", "230", "222", "216", "227", "223"]:
                #     dataY_tmp = "200"
                # else:
                #     dataY_tmp = label
                # line["category"] = dataY_tmp
                # _data = self._preline(line)
                _data = line["pre_clean_text"]
                if _data.strip():
                    labels = classifier.predict([_data])
                    pred_label = labels[0][0][0].replace("__label__", "")
                    pred_prob = labels[0][0][1]
                    if pred_label == "200":
                        sub_label = sub_classifier.predict_proba([_data])
                        pred_label = sub_label[0][0][0].replace("__label__", "")
                        pred_prob = sub_label[0][0][1]

                    line['predict_category'] = pred_label
                    # print(line['predict_top_category'])
                    line['predict_category_proba'] = pred_prob
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
        title = line_json["article_title"]
        content = line_json["text"]
        tag_list = ",".join(line_json["name"])
        dataX = self.clean_content(title + ' ' + content + " " + tag_list)  # 清洗数据
        # dataX = self.clean_title(title)  # 清洗数据
        if dataX:
            # feature_words = self.fe.predict_feature(title)
            feature_words = ""
            new_text = "{} {}".format(dataX, feature_words).strip()
            _data = new_text
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

    def clean_content(self, text):
        text = text.replace("\r", " ").replace("\n", " ").replace("\t", " ")
        text = text.lower()
        cd_instance = CleanDoc(text)
        no_emoji = cd_instance.remove_emoji(text)
        no_url = cd_instance.clean_html(no_emoji)
        no_mail = cd_instance.clean_mail(no_url)
        no_symbol = cd_instance.remove_symbol(no_mail)
        text = re.sub(r"\s+", " ", no_symbol)
        return text

    def evaluate_model(self, datapath, key_, labels=None):
        em = EvaluateModel(datapath, key_name=key_, logger=self.log, label_names=labels)
        return em.evaluate_model_v2()



if __name__ == '__main__':
    s = time.time()
    dataDir = "/data/browser_category/train_v7"
    feature_model = "/data/browser_category/category_feature_words/category_feature"
    # dataDir = "/data/emotion_analysis/taste_ft_model"
    # DataSet(dataDir, feature_model, logger=log)
    bcm = BrowserCategoryModel(dataDir, feature_model, logger=log)
    bcm.train_model()
    e = time.time()
    print('训练浏览器分类模型耗时{}'.format(e - s))