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


class Predictor(object):

    def __init__(self, config, model, logger=None):

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.config = config

        self.model = model
        self.data_obj = None

        # 加载数据集
        self.data_obj = DatasetLoader(config, logger=self.log)

        self.word2index, self.label2index = self.data_obj.word2index, self.data_obj.label2index
        self.index2label = {value: key for key, value in self.label2index.items()}
        self.vocab_size = len(self.word2index)
        self.sequence_length = self.config.sequence_length

        self.save_path = os.path.join(self.config.ckpt_model_path, "{}.ckpt".format(self.config.model_name))
        self.load_model()

    def load_model(self):

        if os.path.exists(self.save_path):
            self.log.info('Reloading model parameters..')
            self.model.load_state_dict(torch.load(self.save_path))
            self.model.eval()
            torch.cuda.empty_cache()
            self.model.to(self.config.device)
            self.log.info("{} loaded!".format(self.save_path))
        else:
            raise ValueError('No such file:[{}]'.format(self.save_path))


    def sentence_to_idx(self, sentence):
        """
        将分词后的句子转换成idx表示
        :param sentence:
        :return:
        """
        sentence_ids = [self.word2index.get(token, self.word2index["<UNK>"]) for token in sentence]
        sentence_pad = sentence_ids[: self.sequence_length] if len(sentence_ids) > self.sequence_length \
            else sentence_ids + [0] * (self.sequence_length - len(sentence_ids))
        return sentence_pad

    def predict(self, sentence):
        """
        给定分词后的句子，预测其分类结果
        :param sentence:
        :return:
        """
        sentence_ids = self.sentence_to_idx(sentence)

        outputs = self.model.forward(sentence_ids)
        prediction = torch.max(outputs.data, 1)[1].cpu()
        label = self.index2label[prediction]
        return label


    def predict_batch(self, sentences):

        sentences_ids = list()
        for sentence in sentences:
            sentence_ids = self.sentence_to_idx(sentence)
            sentences_ids.append(sentence_ids)

        input_ids = torch.LongTensor(sentences_ids).to(self.config.device)
        outputs = self.model.forward((input_ids, None))
        predictions = torch.max(outputs.data, 1)[1].cpu().numpy()
        labels = [self.index2label[pre] for pre in predictions]
        return labels


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


def predict_to_file():
    """
    预测验证
    :return:
    """
    import time
    model_name = "textcnn"
    x = import_module('model_pytorch.{}_model'.format(model_name))
    conf_file = os.path.join(CONFIG_PATH, "{}_pytorch.ini".format(model_name))
    config = x.Config(conf_file, section="THUC_NEWS")

    log_file = os.path.join(config.output_path, '{}_predict_log'.format(config.model_name))
    log = Logger("train_log", log2console=True, log2file=True, logfile=log_file).get_logger()
    log.info("*** Init all params ***")
    log.info(json.dumps(config.all_params, indent=4))

    model = x.Model(config).to(config.device)
    predictor = Predictor(config, model, logger=log)
    files = [os.path.join(config.data_path, "thuc_news.{}.txt".format(i)) for i in ["train", "eval", "test"]]
    # files = [os.path.join(config.data_path, "thuc_news.{}.txt".format(i)) for i in ["test"]]
    predict_file = os.path.join(config.output_path, "thuc_news.predict.txt")
    e = time.time()
    batch_size = 128
    with open(predict_file, "w", encoding="utf-8") as wf:
        for file in files:
            file_type = os.path.split(file)[1].split(".")[1]
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()

                num_batches = len(lines) // batch_size
                for i in range(num_batches + 1):
                    if i+1 % 100 == 0:
                        log.info("已处理{}条".format((i+1) * batch_size))

                    start = i * batch_size
                    end = start + batch_size
                    text_batch = list()
                    true_labels = list()
                    ids = list()
                    if end > len(lines):
                        _lines = lines[start:]
                    else:
                        _lines = lines[start:end]

                    for i, _line in enumerate(_lines):
                        line = json.loads(_line.strip())
                        ids.append(start + i)
                        true_labels.append(line["label"])
                        text_batch.append(line["text"])

                    predict_labels = predictor.predict_batch(text_batch)
                    for j, _ in enumerate(predict_labels):
                        out = dict()
                        out["guid"] = "{}-{}".format(file_type, ids[j])
                        out["true_label"] = true_labels[j]
                        out["predict_label"] = predict_labels[j]
                        if out:
                            wf.write(json.dumps(out, ensure_ascii=False) + "\n")
                            wf.flush()



                # 预测单条
                # for i, _line in enumerate(lines):
                #     line = json.loads(_line.strip())
                #     out = dict()
                #     out["guid"] = "{}-{}".format(file_type, i)
                #     out["true_label"] = line["label"]
                #     out["predict_label"] = predictor.predict(line["text"])
                #     if out:
                #         wf.write(json.dumps(out, ensure_ascii=False) + "\n")
    s = time.time()
    log.info("*** 预测完成")
    log.info("预测耗时: {}s".format(s - e))
    log.info("*** 预测结果评估 ***")
    predict_report(predict_file, log)
    log.info("*** 评估完成 ***")

def predict_report(file, log):
    """
    预测结果评估
    :return:
    """
    from sklearn.metrics import classification_report
    from evaluate.metrics import get_multi_metrics
    import json
    # file = "/data/work/dl_project/data/corpus/thuc_news/thuc_news.predict.txt"
    result = dict()
    result["train"] = (list(), list())
    result["eval"] = (list(), list())
    result["test"] = (list(), list())
    with open(file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for _line in lines:
            line = json.loads(_line.strip())
            guid = line["guid"]
            mode = guid.split("-")[0]
            true_y = line["true_label"]
            pred_y = line["predict_label"]
            if mode == "train":
                result["train"][0].append(true_y)
                result["train"][1].append(pred_y)
            elif mode == "eval":
                result["eval"][0].append(true_y)
                result["eval"][1].append(pred_y)
            elif mode == "test":
                result["test"][0].append(true_y)
                result["test"][1].append(pred_y)

    labels_list = ['财经', '彩票', '房产', '股票', '家居', '教育', '科技',
                   '社会', '时尚', '时政', '体育', '星座', '游戏', '娱乐']
    for k, v in result.items():
        log.info("{}的整体性能:".format(k))
        acc, recall, F1 = get_multi_metrics(v[0], v[1])
        log.info('\n----模型整体 ----\nacc_score:\t{} \nrecall:\t{} \nf1_score:\t{} '.format(acc, recall, F1))
        log.info("{}的详细结果:".format(k))
        class_report = classification_report(v[0], v[1], labels=labels_list)
        log.info('\n----结果报告 ---:\n{}'.format(class_report))






def main():
    # train_model()
    predict_to_file()

if __name__ == "__main__":
    main()