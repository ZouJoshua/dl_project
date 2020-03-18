#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/13/20 5:55 PM
@File    : run_classifier_with_bert_tf.py
@Desc    : 

"""


import os
import json
import argparse
import configparser
import time
import logging
import tensorflow as tf
from model_tensorflow.bert_model import modeling
from nlp_tasks.text_classification.thuc_news.bert_tf_model import BertClassifier
from nlp_tasks.text_classification.thuc_news.dataset_loader_for_bert_tf import DatasetLoader
from nlp_tasks.text_classification.thuc_news.metrics import mean, get_multi_metrics
from setting import DATA_PATH, CONFIG_PATH
from utils.logger import Logger

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "100"



class Trainer(object):
    def __init__(self, config_file, logger=None):

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("bert_train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.config = self.init_config(config_file)

        self.bert_init_checkpoint = self.config.get("init_checkpoint")

        # 加载数据集
        self.data_obj = DatasetLoader(self.config, logger=self.log)
        self.train_features = self.load_data(self.data_obj, mode="train")
        self.log.info("train data size: {}".format(len(self.train_features)))
        self.eval_features = self.load_data(self.data_obj, mode="eval")
        self.log.info("eval data size: {}".format(len(self.eval_features)))

        # 加载label
        self.label_map = self.data_obj.label_map
        self.label_list = [value for key, value in self.label_map.items()]
        self.log.info("label numbers: {}".format(len(self.label_list)))

        self.train_epochs = self.config.getint("num_train_epochs")
        self.train_batch_size = self.config.getint("train_batch_size")
        self.eval_batch_size = self.config.getint("eval_batch_size")
        warmup_proportion = self.config.getfloat("warmup_proportion")
        num_train_steps = int(
            len(self.train_features) / self.train_batch_size * self.train_epochs)
        num_warmup_steps = int(num_train_steps * warmup_proportion)

        # 初始化模型对象
        self.model = self.create_model(num_train_steps, num_warmup_steps)

    def init_config(self, config_file):
        config_ = configparser.ConfigParser()
        config_.read(config_file)
        all_items = config_.items("THUC_NEWS")
        params = {}
        for i in all_items:
            params[i[0]] = i[1]
        self.log.info("*** Init all params ***")
        self.log.info(json.dumps(params, indent=4))

        config = config_["THUC_NEWS"]
        if not config:
            raise Exception("Config file error.")
        return config

    def init_config_v1(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--config_path", help="config path of model")
        args = parser.parse_args()
        with open(args.config_path, "r") as fr:
            config = json.load(fr)
        return config

    def load_data(self, data_obj, mode=None):
        """
        创建数据对象
        :return:
        """
        # 生成训练集对象并生成训练数据

        data_file = os.path.join(self.config.get("data_dir"), "thuc_news.{}.txt".format(mode))
        pkl_file = os.path.join(self.config.get("data_dir"), "thuc_news.{}.pkl".format(mode))
        if not os.path.exists(data_file):
            raise FileNotFoundError

        features = data_obj.gen_data(data_file, pkl_file, mode=mode)

        return features

    def create_model(self, num_train_step, num_warmup_step):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        model = BertClassifier(config=self.config, num_train_step=num_train_step, num_warmup_step=num_warmup_step, logger=self.log)
        return model

    def train(self):
        with tf.Session() as sess:
            tvars = tf.trainable_variables()
            (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(
                tvars, self.bert_init_checkpoint)
            self.log.info("*** Init bert model params ***")
            tf.train.init_from_checkpoint(self.bert_init_checkpoint, assignment_map)
            # self.log.info("*** Init bert model params done ***")
            self.log.info("*** Trainable Variables ***")
            for var in tvars:
                init_string = ""
                if var.name in initialized_variable_names:
                    init_string = ", *INIT_FROM_CKPT*"
                self.log.info("  name = %s, shape = %s%s", var.name, var.shape,
                                init_string)

            sess.run(tf.variables_initializer(tf.global_variables()))

            checkpoint_every = self.config.getint("checkpoint_every")
            output_dir = self.config.get("output_dir")
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)


            current_step = 0
            start = time.time()
            for epoch in range(self.train_epochs):
                self.log.info("----- Epoch {}/{} -----".format(epoch + 1, self.train_epochs))

                for batch in self.data_obj.next_batch(self.train_features, self.train_batch_size, mode="train"):
                    loss, predictions = self.model.train(sess, batch)

                    acc, recall, prec, f_beta = get_multi_metrics(pred_y=predictions, true_y=batch["label_ids"],
                                                                  labels=self.label_list)
                    self.log.info("train-step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                        current_step, loss, acc, recall, prec, f_beta))

                    current_step += 1
                    if self.data_obj and current_step % checkpoint_every == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_aucs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.data_obj.next_batch(self.eval_features, self.eval_batch_size, mode="eval"):
                            eval_loss, eval_predictions = self.model.eval(sess, eval_batch)

                            eval_losses.append(eval_loss)

                            acc, recall, prec, f_beta = get_multi_metrics(pred_y=eval_predictions,
                                                                          true_y=eval_batch["label_ids"],
                                                                          labels=self.label_list)
                            eval_accs.append(acc)
                            eval_recalls.append(recall)
                            eval_precs.append(prec)
                            eval_f_betas.append(f_beta)
                        self.log.info("\n")
                        self.log.info("eval:  loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            mean(eval_losses), mean(eval_accs), mean(eval_aucs), mean(eval_recalls),
                            mean(eval_precs), mean(eval_f_betas)))
                        self.log.info("\n")


                        model_save_path = os.path.join(output_dir, self.config.get("model_name"))
                        self.model.saver.save(sess, model_save_path, global_step=current_step)

            end = time.time()
            self.log.info("total train time: ", end - start)




class Predictor(object):
    def __init__(self, config_file, logger=None):

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("bert_train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.config = self.init_config(config_file)

        self.data_obj = DatasetLoader(self.config, logger=self.log)
        self.label_map = self.data_obj.label_map
        self.index_to_label = {value: key for key, value in self.label_map.items()}
        self.ckpt_model_path = self.config.get("output_dir")
        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def init_config(self, config_file):
        config_ = configparser.ConfigParser()
        config_.read(config_file)
        all_items = config_.items("THUC_NEWS")
        params = {}
        for i in all_items:
            params[i[0]] = i[1]
        self.log.info("*** Init all params ***")
        self.log.info(json.dumps(params, indent=4))

        config = config_["THUC_NEWS"]
        if not config:
            raise Exception("Config file error.")
        return config


    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        ckpt = tf.train.get_checkpoint_state(self.ckpt_model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.log.info('*** Reloading model parameters ***')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(self.ckpt_model_path))

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        self.model = BertClassifier(config=self.config, is_training=False, logger=self.log)

    def predict(self, text):
        """
        给定分词后的句子，预测其分类结果
        :param text:
        :return:
        """
        guid = "predict-1"
        feature = self.data_obj.convert_single_example_to_feature(guid, text)
        input_ids = [feature.input_ids]
        input_masks = [feature.input_masks]
        segment_ids = [feature.segment_ids]
        prediction = self.model.infer(self.sess,
                                      dict(input_ids=input_ids,
                                           input_masks=input_masks,
                                           segment_ids=segment_ids)).tolist()[0]
        label = self.index_to_label[prediction]
        return label





def main():
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'bert_train_log')
    log = Logger("bert_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    conf_file = os.path.join(CONFIG_PATH, "bert_model_config.ini")
    trainer = Trainer(conf_file, logger=log)
    trainer.train()



if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    main()