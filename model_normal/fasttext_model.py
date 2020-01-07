#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-17 上午9:47
@File    : fasttext_model.py
@Desc    : 
"""



import fasttext
import os
import json
import logging

try:
    import configparser
except:
    from six.moves import configparser


class FastTextClassifier:
    """
    利用fasttext来对文本进行分类
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
            self.log.setLevel(logging.INFO)

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
        args_dict["input"] = self.train_path
        # args_dict["model"] = config.get(section, "model")
        args_dict["label"] = config.get(section, "label")
        args_dict["epoch"] = config.getint(section, "epoch")
        # args_dict["silent"] = config.getboolean(section, "silent")
        args_dict["lr"] = config.getfloat(section, "lr")
        args_dict["minn"] = config.getint(section, "minn")
        args_dict["maxn"] = config.getint(section, "maxn")
        args_dict["loss"] = config.get(section, "loss")
        args_dict["min_count"] = config.getint(section, "min_count")
        args_dict["word_ngrams"] = config.getint(section, "word_ngrams")
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


