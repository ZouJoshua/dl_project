#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-17 上午9:47
@File    : fasttext_model.py
@Desc    : 
"""


import os
import json
import logging

import fasttext


try:
    import configparser
except:
    from six.moves import configparser


class Config(object):
    def __init__(self, config_file, section=None):
        config_ = configparser.ConfigParser()
        config_.read(config_file, encoding='utf-8')
        if not section:
            section = "DEFAULT"
        self.all_params = {}
        self.args_list = config_.options(section)

        config = config_[section]
        if not config:
            raise Exception("Config file error.")

        self.all_params["input"] = config.get("input")
        self.all_params["label"] = config.get("label", "__label__")
        self.all_params["epoch"] = config.getint("epoch", 5)
        self.all_params["dim"] = config.getint("dim", 100)
        self.all_params["lr"] = config.getfloat("lr", 0.05)
        self.all_params["lrUpdateRate"] = config.getint("lrUpdateRate", 100)
        self.all_params["ws"] = config.getint("ws", 5)
        self.all_params["minn"] = config.getint("minn", 0)
        self.all_params["maxn"] = config.getint("maxn", 0)
        self.all_params["loss"] = config.get("loss", "softmax")
        self.all_params["minCount"] = config.getint("minCount", 1)
        self.all_params["wordNgrams"] = config.getint("wordNgrams", 1)
        self.all_params["bucket"] = config.getint("bucket", 2000)
        self.all_params["thread"] = config.getint("thread", 12)
        self.all_params["minCountLabel"] = config.getint("minCountLabel", 0)
        self.all_params["neg"] = config.getint("neg")
        self.all_params["t"] = config.getfloat("t", 1e-4)
        self.all_params["verbose"] = config.getint("verbose", 2)
        self.all_params["pretrainedVectors"] = config.get("pretrainedVectors", "")



class FastTextClassifier:
    """
    利用fasttext-0.9.1来对文本进行分类
    """

    def __init__(self, model_path, config, train=False, model_name="",
                 data_path=None, name_mark="", logger=None):
        """
        初始化
        :param file_path: 训练数据路径
        :param model_path: 模型保存路径
        """
        self.args = config.all_params
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("fasttext_train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        if name_mark:
            self.model_file = os.path.join(model_path, "{}classification.bin".format(name_mark))
        else:
            if model_name:
                self.model_file = os.path.join(model_path, model_name)
            else:
                self.model_file = os.path.join(model_path, "classification.bin")


        if not train:
            self.model = self.load(self.model_file)
            assert self.model is not None, '训练模型无法获取'
        else:
            assert data_path is not None, '训练时, file_path不能为None'
            self.train_file = os.path.join(data_path, '{}train.txt'.format(name_mark))
            self.eval_file = os.path.join(data_path, '{}eval.txt'.format(name_mark))
            self.test_file = os.path.join(data_path, '{}test.txt'.format(name_mark))
            if not self.args["input"]:
                self.args["input"] = self.train_file
            self.model = self.train()
            self.save(quantize=False)

    def train(self, detail=True):
        """
        训练:参数可以针对性修改,进行调优
        """

        model = fasttext.train_supervised(**self.args)

        # 训练集
        train_result = model.test(self.train_file)
        self.log.info('训练集准确率： \n样本数N\t{}\n精确率P\t{:.3f}\n召回率R\t{:.3f}'.format(*train_result))
        if detail:
            train_result_detail = model.test_label(self.train_file)
            self.log.info('训练集各类别准确率： {}'.format(json.dumps(train_result_detail, ensure_ascii=False, indent=4)))

        # 验证集
        if os.path.exists(self.eval_file):
            eval_result = model.test(self.eval_file)
            self.log.info('验证集准确率： \n样本数N\t{}\n精确率P\t{:.3f}\n召回率R\t{:.3f}'.format(*eval_result))
            if detail:
                eval_result_detail = model.test_label(self.eval_file)
                self.log.info('验证集各类别准确率： {}'.format(json.dumps(eval_result_detail, ensure_ascii=False, indent=4)))

        # 测试集
        if os.path.exists(self.test_file):
            test_result = model.test(self.test_file)
            self.log.info('测试集准确率: \n样本数N\t{}\n精确率P\t{:.3f}\n召回率R\t{:.3f}'.format(*test_result))
            if detail:
                test_result_detail = model.test_label(self.test_file)
                self.log.info('测试集各类别准确率: {}'.format(json.dumps(test_result_detail, ensure_ascii=False, indent=4)))

        return model

    def save(self, quantize=False):
        self.model.save_model(self.model_file)
        if quantize:
            model = fasttext.load_model(self.model_file)
            model.quantize(input=self.train_file, retrain=True)
            ftz_model = os.path.splitext(self.model_file)[0] + ".ftz"
            model.save_model(ftz_model)

    def predict(self, text, k=1):
        """
        预测一条数据,由于fasttext获取的参数是列表,如果只是简单输入字符串,会将字符串按空格拆分组成列表进行推理
        :param text: 待分类的数据
        :return: 分类后的结果
        """
        if isinstance(text, list):
            output = self.model.predict(text, k=k)
        else:
            output = self.model.predict([text], k=k)
        # print('predict:', output)
        return output

    def load(self, model_file):
        """
        加载训练好的模型
        :param model_path: 训练好的模型路径
        :return:
        """
        if os.path.exists(model_file):
            return fasttext.load_model(model_file)
        else:
            raise Exception("Model file {} not found.".format(model_file))

    @staticmethod
    def print_results(N, p, r):
        print("样本数N\t" + str(N))
        print("精确率P@{}\t {:.3f}".format(1, p))
        print("召回率R@{}\t {:.3f}".format(1, r))




class FastTextClassifierV1(object):
    """
    利用fasttext-0.8.3来对文本进行分类
    """

    def __init__(self, model_path, args_config_file,
                 args_section, train=False, model_name="classification.bin",
                 data_path=None, logger=None):
        """
        初始化
        :param file_path: 训练数据路径
        :param model_path: 模型保存路径
        """
        self.config_file = args_config_file
        self.args_section = args_section
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("fasttext_train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.model_file = os.path.join(model_path, model_name)
        if not train:
            self.model = self.load(self.model_file)
            assert self.model is not None, '训练模型无法获取'
        else:
            assert data_path is not None, '训练时, file_path不能为None'
            self.train_path = os.path.join(data_path, 'train.txt')
            self.test_path = os.path.join(data_path, 'test.txt')
            self.model = self.train()

    def train(self):
        """
        训练:参数可以针对性修改,进行调优
        """
        args_dict = self.get_train_args(section=self.args_section)
        model = fasttext.train_supervised(**args_dict)
        # model.print_results(*model.test(valid_data))
        train_result = model.test(self.train_path)
        self.log.info('训练集准确率： {}'.format(train_result.precision))
        test_result = model.test(self.test_path)
        self.log.info('测试集准确率: {}'.format(test_result.precision))
        return model

    def get_train_args(self, section=None):
        config = configparser.ConfigParser()
        config.read(self.config_file, encoding='utf-8')
        if not section:
            section = "fasttext.args"
        args_list = config.options(section)
        self.log.info("配置文件参数列表：{}".format(args_list))
        args_dict = dict()
        # args_dict["model"] = config.get(section, "model")
        args_dict["label"] = config.get(section, "label")
        args_dict["epoch"] = config.getint(section, "epoch")
        # args_dict["silent"] = config.getboolean(section, "silent")
        args_dict["lr"] = config.getfloat(section, "lr")
        args_dict["loss"] = config.get(section, "loss")

        # fasttext-0.8.3的参数
        args_dict["input_file"] = self.train_path
        args_dict["lr_update_rate"] = config.getint(section, "lr_update_rate")
        args_dict["min_count"] = config.getint(section, "min_count")
        args_dict["word_ngrams"] = config.getint(section, "word_ngrams")
        args_dict["minCount"] = config.getint(section, "minCount")
        args_dict["wordNgrams"] = config.getint(section, "wordNgrams")
        args_dict["bucket"] = config.getint(section, "bucket")
        self.log.info("训练参数：{}".format(json.dumps(args_dict, indent=4)))
        return args_dict



    def predict(self, text, k=1):
        """
        预测一条数据,由于fasttext获取的参数是列表,如果只是简单输入字符串,会将字符串按空格拆分组成列表进行推理
        :param text: 待分类的数据
        :return: 分类后的结果
        """
        if isinstance(text, list):
            output = self.model.predict_proba(text, k=k)
        else:
            output = self.model.predict_proba([text], k=k)
        # print('predict:', output)
        return output

    def load(self, model_file):
        """
        加载训练好的模型
        :param model_path: 训练好的模型路径
        :return:
        """
        if os.path.exists(model_file):
            return fasttext.load_model(model_file)
        else:
            raise Exception("Model file {} not found.".format(model_file))
