#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/23/20 4:39 PM
@File    : run_classifier_multi_models.py
@Desc    : 

"""


import torch.nn as nn
import torch.nn.functional as F
from sklearn import metrics
import time
import torch
import numpy as np
import os
from importlib import import_module
from nlp_tasks.text_classification.thuc_news.dataset_loader_for_multi_models_pt import get_time_dif, DatasetLoader
from tensorboardX import SummaryWriter
from utils.logger import Logger
from setting import CONFIG_PATH
import json
import logging

class Trainer(object):
    def __init__(self, config, logger=None):
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.config = config

        self.model = None
        self.data_obj = None

        # 加载数据集
        self.data_obj = DatasetLoader(config, logger=self.log)
        self.label2index = self.data_obj.label2index
        self.word_embedding = self.data_obj.word_embedding
        self.label_list = [kv[0] for kv in sorted(self.label2index.items(), key=lambda item: item[1])]

        self.train_iter, self.train_size = self.load_data("train")
        self.log.info("*** Train data size: {} ***".format(self.train_size))
        self.vocab_size = self.data_obj.vocab_size
        self.log.info("*** Vocab size: {} ***".format(self.vocab_size))

        self.eval_iter, self.eval_size = self.load_data("eval")
        self.log.info("*** Eval data size: {} ***".format(self.eval_size))
        self.log.info("Label numbers: {}".format(len(self.label_list)))

        # if self.config.model_name != 'transformer_pytorch':
        #     self.init_network(self.model)
        self.save_path = os.path.join(self.config.ckpt_model_path, "{}.ckpt".format(self.config.model_name))


    def load_data(self, mode):

        """
        创建数据对象
        :return:
        """
        data_file = os.path.join(self.config.data_path, "thuc_news.{}.txt".format(mode))
        pkl_file = os.path.join(self.config.data_path, "{}_data_pt_{}.pkl".format(mode, self.config.sequence_length))
        if not os.path.exists(data_file):
            raise FileNotFoundError
        input_data = self.data_obj.build_dataset(data_file, pkl_file, mode)
        data_iter = self.data_obj.build_iterator(input_data)
        return data_iter, len(input_data)

    # 权重初始化，默认xavier
    def init_network(self, model, method='xavier', exclude='embedding', seed=123):
        for name, w in model.named_parameters():
            if exclude not in name:
                if 'weight' in name:
                    if method == 'xavier':
                        nn.init.xavier_normal_(w)
                    elif method == 'kaiming':
                        nn.init.kaiming_normal_(w)
                    else:
                        nn.init.normal_(w)
                elif 'bias' in name:
                    nn.init.constant_(w, 0)
                else:
                    pass


    def train(self, model):
        self.log.info("*** 模型结构:{} ***".format(model.parameters))
        start_time = time.time()
        optimizer = torch.optim.Adam(model.parameters(), lr=self.config.learning_rate)

        model.train()
        # 学习率指数衰减，每次epoch：学习率 = gamma * 学习率
        # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
        total_batch = 0  # 记录进行到多少batch
        dev_best_loss = float('inf')
        last_improve = 0  # 记录上次验证集loss下降的batch数
        flag = False  # 记录是否很久没有效果提升
        train_summary_path = os.path.join(self.config.output_path, "summary")
        if not os.path.exists(train_summary_path):
            os.makedirs(train_summary_path)
        log_file = os.path.join(train_summary_path, time.strftime('%m-%d_%H.%M', time.localtime()))
        writer = SummaryWriter(log_dir=log_file)
        for epoch in range(self.config.num_epochs):
            self.log.info("----- Epoch {}/{} -----".format(epoch + 1, self.config.num_epochs))
            # scheduler.step() # 学习率衰减
            for i, (trains, labels) in enumerate(self.train_iter):
                total_batch += 1
                # self.log.info("数据:{}".format(trains))
                # self.log.info("label:{}".format(labels))
                optimizer.zero_grad()
                outputs = model.forward(trains)
                loss = F.cross_entropy(outputs, labels)
                loss.backward()
                optimizer.step()
                # self.log.info("train-step:{} Loss:{}".format(total_batch, loss))

                if total_batch % self.config.eval_every_step == 0:
                    # 每多少轮输出在训练集和验证集上的效果
                    true = labels.data.cpu()
                    predic = torch.max(outputs.data, 1)[1].cpu()
                    train_acc = metrics.accuracy_score(true, predic)
                    dev_acc, dev_loss = self.evaluate(model, self.eval_iter)
                    if dev_loss < dev_best_loss:
                        dev_best_loss = dev_loss
                        torch.save(model.state_dict(), self.save_path)
                        improve = '*'
                        last_improve = total_batch
                    else:
                        improve = ''
                    time_dif = get_time_dif(start_time)
                    msg = "Iter: {0:>6},  Train Loss: {1:>5.2},  Train Acc: {2:>6.2%},  Val Loss: {3:>5.2},  Val Acc: {4:>6.2%},  Time: {5} {6}"
                    self.log.info(msg.format(total_batch, loss.item(), train_acc, dev_loss, dev_acc, time_dif, improve))
                    writer.add_scalar("loss/train", loss.item(), total_batch)
                    writer.add_scalar("loss/dev", dev_loss, total_batch)
                    writer.add_scalar("acc/train", train_acc, total_batch)
                    writer.add_scalar("acc/dev", dev_acc, total_batch)
                    # model.train()
                    if total_batch - last_improve > self.config.require_improvement:
                        # 验证集loss超过10个batch没下降，结束训练
                        self.log.info("No optimization for a long time, auto-stopping...")
                        flag = True
                        break
            if flag:
                break
        writer.close()
        self.test(model, self.eval_iter)


    def test(self, model, test_iter):
        # test

        model.load_state_dict(torch.load(self.save_path))
        model.eval()
        start_time = time.time()
        test_acc, test_loss, test_report, test_confusion = self.evaluate(model, test_iter, test=True)
        msg = 'Test Loss: {0:>5.2},  Test Acc: {1:>6.2%}'
        self.log.info(msg.format(test_loss, test_acc))
        self.log.info("Precision, Recall and F1-Score...")
        self.log.info("\n{}".format(test_report))
        self.log.info("Confusion Matrix...")
        self.log.info("\n{}".format(test_confusion))
        time_dif = get_time_dif(start_time)
        self.log.info("Time usage:{}".format(time_dif))


    def evaluate(self, model, data_iter, test=False):
        model.eval()
        loss_total = 0
        predict_all = np.array([], dtype=int)
        labels_all = np.array([], dtype=int)
        with torch.no_grad():
            for texts, labels in self.eval_iter:
                outputs = model(texts)
                loss = F.cross_entropy(outputs, labels)
                loss_total += loss
                labels = labels.data.cpu().numpy()
                predic = torch.max(outputs.data, 1)[1].cpu().numpy()
                labels_all = np.append(labels_all, labels)
                predict_all = np.append(predict_all, predic)

        acc = metrics.accuracy_score(labels_all, predict_all)
        if test:
            report = metrics.classification_report(labels_all, predict_all, target_names=self.label_list, digits=4)
            confusion = metrics.confusion_matrix(labels_all, predict_all)
            return acc, loss_total / len(data_iter), report, confusion
        return acc, loss_total / len(data_iter)


def train_model():
    """
    :return:
    """

    # 搜狗新闻:embedding_SougouNews.npz, 腾讯:embedding_Tencent.npz, 随机初始化:random
    # embedding = 'embedding_SougouNews.npz'
    # if args.embedding == 'random':
    #     embedding = 'random'
    # model_name = args.model
    # if model_name == 'FastText':
    #     from utils_fasttext import build_dataset, build_iterator, get_time_dif
    #     embedding = 'random'
    # else:
    #     from utils import build_dataset, build_iterator, get_time_dif

    # textcnn, textrnn, fasttext, textrcnn, textrnn_attention, dpcnn, transformer
    model_name = "textcnn"
    x = import_module('model_pytorch.{}_model'.format(model_name))
    conf_file = os.path.join(CONFIG_PATH, "{}_pytorch.ini".format(model_name))
    config = x.Config(conf_file, section="THUC_NEWS")
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    output = config.output_path
    if not os.path.exists(output):
        os.makedirs(output)
    log_file = os.path.join(output, '{}_train_log'.format(config.model_name))
    log = Logger("train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    log.info("*** Init all params ***")
    log.info(json.dumps(config.all_params, indent=4))
    # train
    trainer = Trainer(config, logger=log)
    # print(trainer.word_embedding.shape)
    model = x.Model(config, pretrain_embedding=trainer.word_embedding).to(config.device)
    if model_name != 'Transformer':
        trainer.init_network(model)
    trainer.train(model)




def main():
    train_model()


if __name__ == "__main__":
    main()