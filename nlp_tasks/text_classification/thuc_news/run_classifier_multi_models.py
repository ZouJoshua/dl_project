#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/23/20 4:39 PM
@File    : run_classifier_multi_models.py
@Desc    : 

"""


import json
import os
import argparse
import pickle

import tensorflow as tf
from model_tensorflow.basic_train import TrainerBase
from model_tensorflow.basic_predict import PredictorBase
from nlp_tasks.text_classification.thuc_news.dataset_loader_for_multi_models import DatasetLoader
from model_tensorflow.textcnn_model import TextCNN, Config
from evaluate.metrics import get_binary_metrics, get_multi_metrics, mean, get_custom_multi_metrics

import logging
from utils.logger import Logger
from setting import CONFIG_PATH


class Trainer(TrainerBase):
    def __init__(self, config, logger=None):
        super(Trainer, self).__init__()

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.config = config

        self.data_obj = None
        self.model = None
        # self.builder = tf.saved_model.builder.SavedModelBuilder("../pb_model/textcnn/bilstm/savedModel")

        # 加载数据集
        self.data_obj = DatasetLoader(config, logger=self.log)
        self.label2index = self.data_obj.label2index
        self.word_embedding = self.data_obj.word_embedding
        self.label_list = [value for key, value in self.label2index.items()]

        self.train_inputs, self.train_labels = self.load_data("train")
        self.log.info("*** Train data size: {} ***".format(len(self.train_labels)))
        self.vocab_size = self.data_obj.vocab_size
        self.log.info("*** Vocab size: {} ***".format(self.vocab_size))


        self.eval_inputs, self.eval_labels = self.load_data("eval")
        self.log.info("*** Eval data size: {} ***".format(len(self.eval_labels)))
        self.log.info("Label numbers: {}".format(len(self.label_list)))
        # 初始化模型对象
        self.create_model()

    def load_data(self, mode):
        """
        创建数据对象
        :return:
        """
        data_file = os.path.join(self.config.data_path, "thuc_news.{}.txt".format(mode))
        pkl_file = os.path.join(self.config.data_path, "{}_data.pkl".format(mode))
        if not os.path.exists(data_file):
            raise FileNotFoundError
        inputs, labels = self.data_obj.convert_examples_to_features(data_file, pkl_file, mode)
        return inputs, labels

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config.model_name == "textcnn":
            self.model = TextCNN(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_embedding)
        # elif self.config["model_name"] == "bilstm":
        #     self.model = BiLstmModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        # elif self.config["model_name"] == "bilstm_atten":
        #     self.model = BiLstmAttenModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        # elif self.config["model_name"] == "rcnn":
        #     self.model = RcnnModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        # elif self.config["model_name"] == "transformer":
        #     self.model = TransformerModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)

    def train(self):
        """
        训练模型
        :return:
        """
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.8, allow_growth=True)
        sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())

            # 创建train和eval的summary路径和写入对象
            train_summary_path = os.path.join(self.config.output_path, "summary", "train")
            if not os.path.exists(train_summary_path):
                os.makedirs(train_summary_path)
            train_summary_writer = tf.summary.FileWriter(train_summary_path, sess.graph)

            eval_summary_path = os.path.join(self.config.output_path, "summary", "eval")
            if not os.path.exists(eval_summary_path):
                os.makedirs(eval_summary_path)
            eval_summary_writer = tf.summary.FileWriter(eval_summary_path, sess.graph)

            for epoch in range(self.config.num_epochs):
                self.log.info("----- Epoch {}/{} -----".format(epoch + 1, self.config.num_epochs))

                for batch in self.data_obj.next_batch(self.train_inputs, self.train_labels,
                                                            self.config.train_batch_size):
                    summary, global_step, loss, predictions = self.model.train(sess, batch, self.config.dropout_keep_prob)

                    train_summary_writer.add_summary(summary, global_step=global_step)
                    train_summary_writer.flush()

                    if self.config.num_labels == 1:
                        acc, auc, recall, prec, f_beta = get_binary_metrics(pred_y=predictions, true_y=batch["y"])
                        self.log.info("train-step: {}, loss: {}, acc: {}, auc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            global_step, loss, acc, auc, recall, prec, f_beta))
                    elif self.config.num_labels > 1:
                        acc, recall, prec, f_beta = get_custom_multi_metrics(pred_y=predictions, true_y=batch["y"],
                                                                      labels=self.label_list)
                        self.log.info("train-step: {}, loss: {}, acc: {}, recall: {}, precision: {}, f_beta: {}".format(
                            global_step, loss, acc, recall, prec, f_beta))

                        # acc, recall, F1 = get_multi_metrics(pred_y=predictions, true_y=batch["y"])
                        # self.log.info("train-step: {}, loss: {}, acc: {}, recall: {}, F1_score: {}".format(
                        #     global_step, loss, acc, recall, F1))

                    if self.data_obj and global_step % self.config.eval_every_step == 0:

                        eval_losses = []
                        eval_accs = []
                        eval_aucs = []
                        eval_recalls = []
                        eval_precs = []
                        eval_f_betas = []
                        for eval_batch in self.data_obj.next_batch(self.eval_inputs, self.eval_labels,
                                                                        self.config.eval_batch_size):
                            eval_summary, eval_step, eval_loss, eval_predictions = self.model.eval(sess, eval_batch)
                            eval_summary_writer.add_summary(eval_summary, global_step=eval_step)
                            eval_summary_writer.flush()

                            eval_losses.append(eval_loss)
                            if self.config.num_labels == 1:
                                acc, auc, recall, prec, f_beta = get_binary_metrics(pred_y=eval_predictions,
                                                                                    true_y=eval_batch["y"])
                                eval_accs.append(acc)
                                eval_aucs.append(auc)
                                eval_recalls.append(recall)
                                eval_precs.append(prec)
                                eval_f_betas.append(f_beta)
                            elif self.config.num_labels > 1:
                                acc, recall, prec, f_beta = get_custom_multi_metrics(pred_y=eval_predictions,
                                                                              true_y=eval_batch["y"],
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

                        if self.config.ckpt_model_path:
                            ckpt_model_path = self.config.ckpt_model_path
                        else:
                            ckpt_model_path = os.path.join(self.config.output_path, "model")

                        if not os.path.exists(ckpt_model_path):
                            os.makedirs(ckpt_model_path)
                        model_save_path = os.path.join(ckpt_model_path, self.config.model_name)
                        self.model.saver.save(sess, model_save_path, global_step=global_step)

            # inputs = {"inputs": tf.saved_model.utils.build_tensor_info(self.model.inputs),
            #           "keep_prob": tf.saved_model.utils.build_tensor_info(self.model.keep_prob)}
            #
            # outputs = {"predictions": tf.saved_model.utils.build_tensor_info(self.model.predictions)}
            #
            # # method_name决定了之后的url应该是predict还是classifier或者regress
            # prediction_signature = tf.saved_model.signature_def_utils.build_signature_def(inputs=inputs,
            #                                                                               outputs=outputs,
            #                                                                               method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME)
            # legacy_init_op = tf.group(tf.tables_initializer(), name="legacy_init_op")
            # self.builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING],
            #                                           signature_def_map={"classifier": prediction_signature},
            #                                           legacy_init_op=legacy_init_op)
            #
            # self.builder.save()


class Predictor(PredictorBase):
    def __init__(self, config, logger=None):
        super(Predictor, self).__init__(config)

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.model = None
        self.config = config

        self.word2index, self.label2index = self.load_vocab()
        self.index2label = {value: key for key, value in self.label2index.items()}
        self.vocab_size = len(self.word2index)
        self.word_embedding = None
        self.sequence_length = self.config.sequence_length

        # 创建模型
        self.create_model()
        # 加载计算图
        self.load_graph()

    def load_vocab(self):
        # 将词汇-索引映射表加载出来
        with open(os.path.join(self.config.data_path, "word2index.pkl"), "rb") as f:
            word_to_index = pickle.load(f)

        with open(os.path.join(self.config.data_path, "label2index.pkl"), "rb") as f:
            label_to_index = pickle.load(f)

        return word_to_index, label_to_index

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

    def load_graph(self):
        """
        加载计算图
        :return:
        """
        self.sess = tf.Session()
        if self.config.ckpt_model_path:
            ckpt_model_path = self.config.ckpt_model_path
        else:
            ckpt_model_path = os.path.join(self.config.output_path, "model")

        ckpt = tf.train.get_checkpoint_state(ckpt_model_path)
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            self.log.info('Reloading model parameters..')
            self.model.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            raise ValueError('No such file:[{}]'.format(ckpt_model_path))

    def create_model(self):
        """
        根据config文件选择对应的模型，并初始化
        :return:
        """
        if self.config.model_name == "textcnn":
            self.model = TextCNN(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_embedding)
        # elif self.config["model_name"] == "bilstm":
        #     self.model = BiLstmModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        # elif self.config["model_name"] == "bilstm_atten":
        #     self.model = BiLstmAttenModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        # elif self.config["model_name"] == "rcnn":
        #     self.model = RcnnModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)
        # elif self.config["model_name"] == "transformer":
        #     self.model = TransformerModel(config=self.config, vocab_size=self.vocab_size, word_vectors=self.word_vectors)

    def predict(self, sentence):
        """
        给定分词后的句子，预测其分类结果
        :param sentence:
        :return:
        """
        sentence_ids = self.sentence_to_idx(sentence)

        prediction = self.model.infer(self.sess, [sentence_ids]).tolist()[0]
        label = self.index2label[prediction]
        return label



def textcnn_train_model():
    """
    textcnn训练模型
    :return:
    """
    conf_file = os.path.join(CONFIG_PATH, "textcnn.ini")
    config = Config(conf_file, section="THUC_NEWS")
    output = config.output_path
    if not os.path.exists(output):
        os.makedirs(output)
    log_file = os.path.join(output, 'textcnn_train_log')
    log = Logger("train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    log.info("*** Init all params ***")
    log.info(json.dumps(config.all_params, indent=4))
    trainer = Trainer(config, logger=log)
    trainer.train()

def textcnn_predict_to_file():
    """
    textcnn预测验证
    :return:
    """
    import time
    conf_file = os.path.join(CONFIG_PATH, "textcnn.ini")
    config = Config(conf_file, section="THUC_NEWS")
    output = config.output_path
    log_file = os.path.join(output, 'textcnn_predict_log')

    log = Logger("train_log", log2console=True, log2file=True, logfile=log_file).get_logger()
    log.info("*** Init all params ***")
    log.info(json.dumps(config.all_params, indent=4))
    predictor = Predictor(config, logger=log)
    files = [os.path.join(config.data_path, "thuc_news.{}.txt".format(i)) for i in ["train", "eval", "test"]]
    # files = [os.path.join(config.data_path, "thuc_news.{}.txt".format(i)) for i in ["test"]]
    predict_file = os.path.join(output, "thuc_news.predict.txt")
    e = time.time()
    with open(predict_file, "w", encoding="utf-8") as wf:
        for file in files:
            file_type = os.path.split(file)[1].split(".")[1]
            with open(file, "r", encoding="utf-8") as f:
                lines = f.readlines()
                for i, _line in enumerate(lines):
                    line = json.loads(_line.strip())
                    out = dict()
                    out["guid"] = "{}-{}".format(file_type, i)
                    out["true_label"] = line["label"]
                    out["predict_label"] = predictor.predict(line["text"])
                    if out:
                        wf.write(json.dumps(out, ensure_ascii=False) + "\n")
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
    # textcnn_train_model()
    textcnn_predict_to_file()


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", help="config path of model")
    # args = parser.parse_args()
    # trainer = Trainer(args)
    # trainer.train()
    main()
