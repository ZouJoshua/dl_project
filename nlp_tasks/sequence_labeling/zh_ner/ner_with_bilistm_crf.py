#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 4/20/20 9:14 PM
@File    : ner_with_bilistm_crf.py
@Desc    : 

"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'    # Only device 1 will be seen.
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'    # default: 0

import tensorflow as tf
import json
import time
from model_tensorflow.basic_train import TrainerBase
from model_tensorflow.ner_bilstm_crf_model import NERTagger
from nlp_tasks.sequence_labeling.zh_ner.dataset_loader import DatasetLoader
from nlp_tasks.sequence_labeling.zh_ner.preprocess_data import bioes_to_bio
from model_tensorflow.ner_bilstm_crf_model import Config
from nlp_tasks.sequence_labeling.zh_ner.conlleval import return_report
from evaluate.custom_metrics import mean

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

        # 加载数据集
        self.data_obj = DatasetLoader(config, logger=self.log)
        self.label2index = self.data_obj.label2index
        self.index2label = {v:k for k, v in self.label2index.items()}
        self.word_embedding = self.data_obj.word_embedding
        # self.label_list = [value for key, value in self.label2index.items()]
        self.label_list = [kv[0] for kv in sorted(self.label2index.items(), key=lambda item: item[1])]

        self.vocab_size = self.data_obj.vocab_size
        self.log.info("*** Vocab size: {} ***".format(self.vocab_size))

        self.log.info("*** Label numbers: {} ***".format(len(self.label_list)))
        self.log.info("Label list:{}".format(self.label_list))

        self.train_inputs, self.train_labels = self.load_data("train")
        self.log.info("*** Train data size: {} ***".format(len(self.train_labels)))

        self.eval_inputs, self.eval_labels = self.load_data("test")
        self.log.info("*** Eval data size: {} ***".format(len(self.eval_labels)))

        # self.test_inputs, self.test_labels = self.load_data("test")
        # self.log.info("*** Test data size: {} ***".format(len(self.test_labels)))

        # 初始化模型对象
        self.create_model()



    def create_model(self):
        self.model = NERTagger(self.config, vocab_size=self.vocab_size, word_vectors=self.word_embedding)

    def load_data(self, mode):
        """
        创建数据对象
        :return:
        """
        data_file = os.path.join(self.config.data_path, "{}.txt".format(mode))
        pkl_file = os.path.join(self.config.data_path, "{}_data_{}.pkl".format(mode, self.config.sequence_length))
        if not os.path.exists(data_file):
            raise FileNotFoundError
        inputs, labels = self.data_obj.convert_examples_to_features(data_file, pkl_file, mode)
        return inputs, labels


    def train(self):
        """
        :param train: training data
        :param dev: testing data
        :return:
        """

        # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7, allow_growth=True)
        # sess_config = tf.ConfigProto(log_device_placement=False, allow_soft_placement=True, gpu_options=gpu_options)
        sess_config = tf.ConfigProto(device_count={"CPU": 4}, log_device_placement=False, allow_soft_placement=True)

        with tf.Session(config=sess_config) as sess:
            # 初始化变量值
            sess.run(tf.global_variables_initializer())

            dev_best_loss = float('inf')
            last_improve = 0  # 记录上次验证集loss下降的batch数
            flag = False  # 记录是否很久没有效果提升

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
                                                            self.config.batch_size):

                    summary, global_step, loss, predictions = self.model.train(sess, batch, self.config.dropout_keep_prob)

                    train_summary_writer.add_summary(summary, global_step=global_step)
                    train_summary_writer.flush()
                    time.sleep(0.5)
                    msg = "train-step: {0:>6}, ner_loss:{1:>5.4}"
                    self.log.info(msg.format(global_step, loss))
                    if self.data_obj and global_step % self.config.eval_every_step == 0:
                        self.evaluate(sess, eval_summary_writer, test=False)

                        if self.config.ckpt_model_path:
                            ckpt_model_path = self.config.ckpt_model_path
                        else:
                            ckpt_model_path = os.path.join(self.config.output_path, "model")

                        if not os.path.exists(ckpt_model_path):
                            os.makedirs(ckpt_model_path)
                        model_save_path = os.path.join(ckpt_model_path, self.config.model_name)
                        self.model.saver.save(sess, model_save_path, global_step=global_step)

                #         if global_step - last_improve > self.config.require_improvement:
                #             # 验证集loss超过10个batch没下降，结束训练
                #             self.log.info("No optimization for a long time, auto-stopping...")
                #             flag = True
                #             break
                # if flag:
                #     break

    def evaluate(self, sess, summary_writer, test=False):
        ner_results = list()
        loss_list = list()
        for batch_data in self.data_obj.next_batch(self.eval_inputs, self.eval_labels,
                                                   self.config.batch_size):

            batch_words = batch_data["word_list"]
            batch_tags = batch_data["y"]

            summary, step, loss, lengths, trans, predictions = self.model.eval(sess, batch_data)
            summary_writer.add_summary(summary, global_step=step)
            summary_writer.flush()

            loss_list.append(loss)

            for i in range(len(batch_words)):
                result = []
                word_list = batch_words[i][:lengths[i]]
                gold = bioes_to_bio([self.index2label[int(x)] for x in batch_tags[i][:lengths[i]]])
                pred = bioes_to_bio([self.index2label[int(x)] for x in predictions[i][:lengths[i]]])
                for char, gold, pred in zip(word_list, gold, pred):
                    result.append(" ".join([char, gold, pred]))
                ner_results.append(result)

        eval_lines = self.test_ner(ner_results, self.config.output_path)
        for line in eval_lines:
            self.log.info(line)
        f1 = float(eval_lines[1].strip().split()[-1])
        msg = "eval-step loss:{0:>5.2}, F1_score:{1:>6.2%}"

        self.log.info(msg.format(mean(loss_list), f1/100))
        if test:
            best_test_f1 = self.model.best_test_f1.eval()
            if f1 > best_test_f1:
                tf.assign(self.model.best_test_f1, f1).eval()
                self.log.info('new best test f1 score:{:>.3f}'.format(f1))
            return f1 > best_test_f1

    def test_ner(self, results, path):
        """
        :param results:
        :param path:
        :return:
        """
        output_file = os.path.join(path, 'ner_predict.utf8')
        with open(output_file, "w", encoding="utf-8") as f_write:
            to_write = []
            for line in results:
                for iner_line in line:
                    to_write.append(iner_line + "\n")
                to_write.append("\n")
            f_write.writelines(to_write)
        eval_lines = return_report(output_file)
        return eval_lines





def train_model():
    """
    训练模型
    :return:
    """
    conf_file = os.path.join(CONFIG_PATH, "ner_bilstm_crf.ini")
    config = Config(conf_file, section="ZH_NER_BILSTM_CRF")
    output = config.output_path
    if not os.path.exists(output):
        os.makedirs(output)
    log_file = os.path.join(output, '{}_train_log'.format(config.model_name))
    log = Logger("train_log", log2console=True, log2file=True, logfile=log_file).get_logger()
    log.info("*** Init all params ***")
    log.info(json.dumps(config.all_params, indent=4))
    trainer = Trainer(config, logger=log)
    trainer.train()


def main():
    """
    model_name = <"ner_bilstm_crf">
    :return:
    """
    train_model()


if __name__ == "__main__":
    # 读取用户在命令行输入的信息
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config_path", help="config path of model")
    # args = parser.parse_args()
    # trainer = Trainer(args)
    # trainer.train()
    main()