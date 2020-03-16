#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/11/20 5:47 PM
@File    : train_classifier_v1.py
@Desc    : 

"""
import tensorflow as tf
import configparser
import os
import json

import model_tensorflow.bert_model.modeling as modeling
import model_tensorflow.bert_model.tokenization as tokenization
from model_tensorflow.bert_model.finetuning_processer import PaddingInputExample
from nlp_tasks.text_classification.thuc_news.dataset_loader import BertTFDatasetProcessor
from nlp_tasks.text_classification.thuc_news.thuc_news_classifier_v1 import BertCategoryModel

import logging
from utils.logger import Logger
from setting import CONFIG_PATH, DATA_PATH



class ThucNewsTrainer:


    def __init__(self, config_file, logger=None):

        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("bert_train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.init_params(config_file)
        self.bert_config = modeling.BertConfig.from_json_file(self.bert_config_file)

        if self.max_seq_length > self.bert_config.max_position_embeddings:
            raise ValueError(
                "Cannot use sequence length %d because the BERT model "
                "was only trained up to sequence length %d" %
                (self.max_seq_length, self.bert_config.max_position_embeddings))
        self.processor = BertTFDatasetProcessor()
        self.init_tfrecord(self.processor)
        self.bert_model = BertCategoryModel(self.bert_config, self.num_labels, logger=self.log)
        self.init_estimator()



    def init_params(self, config_file):
        config_ = configparser.ConfigParser()
        config_.read(config_file)
        all_items = config_.items("THUC_NEWS")
        params = {}
        for i in all_items:
            params[i[0]] = i[1]
        self.log.info("***** Init Params *****")
        self.log.info(json.dumps(params, indent=4))

        self.config = config_["THUC_NEWS"]
        if not self.config:
            raise Exception("Config file error.")
        self.data_dir = self.config.get("data_dir")
        self.bert_config_file = self.config.get("bert_config_file")
        self.vocab_file = self.config.get("vocab_file")
        self.output_dir = self.config.get("output_dir")
        self.init_checkpoint = self.config.get("init_checkpoint", None)
        self.do_lower_case = self.config.getboolean("do_lower_case", False)
        self.max_seq_length = self.config.getint("max_seq_length")
        self.train_batch_size = self.config.getint("train_batch_size")
        self.eval_batch_size = self.config.getint("eval_batch_size")
        self.predict_batch_size = self.config.getint("predict_batch_size")
        self.learning_rate = self.config.getfloat("learning_rate")
        self.num_train_epochs = self.config.getfloat("num_train_epochs")
        self.warmup_proportion = self.config.getfloat("warmup_proportion")
        self.save_checkpoints_steps = self.config.getint("save_checkpoints_steps")
        self.iterations_per_loop = self.config.getint("iterations_per_loop")
        self.use_tpu = self.config.getboolean("use_tpu", False)
        self.tpu_name = self.config.get("tpu_name", None)
        self.tpu_zone = self.config.get("tpu_zone", None)
        self.gcp_project = self.config.get("gcp_project", None)
        self.num_tpu_cores = self.config.getint("num_tpu_cores")
        self.master = self.config.get("master", None)



    def init_tfrecord(self, processor):
        self.label_list = processor.get_labels()
        self.num_labels = len(self.label_list)
        tokenizer = tokenization.FullTokenizer(
            vocab_file=self.vocab_file, do_lower_case=self.do_lower_case)

        # 训练集序列化
        self.train_file = os.path.join(self.data_dir, "train.tf_record")
        if not os.path.exists(self.train_file):
            self.log.info("***** Convert Train Examples To TF_Rrecord *****")
            self.train_examples = None
            self.num_train_steps = None
            self.num_warmup_steps = None
            # 返回列表是一行一行的Inputexample对象，每行包括了guid，train_a,label
            train_examples = processor.get_train_examples(self.data_dir)
            self.num_train_examples = len(train_examples)
            # 训练的次数：(训练集的样本数/每批次大小)*训练几轮
            self.num_train_steps = int(
                self.num_train_examples / self.train_batch_size * self.num_train_epochs)
            # 在预热学习中，线性地增加学习率
            self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)
            # 实现Input_example到Feature_example, TFRecord化
            processor.file_based_convert_examples_to_features(
                train_examples, self.label_list, self.max_seq_length, tokenizer, self.train_file)
        else:
            ori_train_file = os.path.join(self.data_dir, "thuc_news.train.txt")
            self.num_train_examples = len(processor._read_json(ori_train_file))
            self.num_train_steps = int(
                self.num_train_examples / self.train_batch_size * self.num_train_epochs)
            self.num_warmup_steps = int(self.num_train_steps * self.warmup_proportion)

        # 验证集序列化
        self.eval_file = os.path.join(self.data_dir, "eval.tf_record")
        if not os.path.exists(self.eval_file):
            self.log.info("***** Convert Eval Examples To TF_Rrecord *****")
            eval_examples = self.processor.get_dev_examples(self.data_dir)
            self.num_actual_eval_examples = len(eval_examples)
            if self.use_tpu:
                # TPU requires a fixed batch size for all batches, therefore the number
                # of examples must be a multiple of the batch size, or else examples
                # will get dropped. So we pad with fake examples which are ignored
                # later on. These do NOT count towards the metric (all tf.metrics
                # support a per-instance weight, and these get a weight of 0.0).
                while len(eval_examples) % self.eval_batch_size != 0:
                    eval_examples.append(PaddingInputExample())

            processor.file_based_convert_examples_to_features(
                eval_examples, self.label_list, self.max_seq_length, tokenizer, self.eval_file)
        else:
            ori_eval_file = os.path.join(self.data_dir, "thuc_news.eval.txt")
            self.num_actual_eval_examples = len(processor._read_json(ori_eval_file))


        # 测试集序列化
        self.predict_file = os.path.join(self.data_dir, "predict.tf_record")
        if not os.path.exists(self.predict_file):
            self.log.info("***** Convert Test Examples To TF_Rrecord *****")
            predict_examples = processor.get_test_examples(self.data_dir)
            self.num_actual_predict_examples = len(predict_examples)
            if self.use_tpu:
                # TPU requires a fixed batch size for all batches, therefore the number
                # of examples must be a multiple of the batch size, or else examples
                # will get dropped. So we pad with fake examples which are ignored
                # later on.
                while len(predict_examples) % self.predict_batch_size != 0:
                    predict_examples.append(PaddingInputExample())

            # 1.调用convert_single_example转化Input_example为Feature_example
            # 2.转换为TFRecord格式，便于大型数据处理
            processor.file_based_convert_examples_to_features(predict_examples, self.label_list,
                                                    self.max_seq_length, tokenizer,
                                                    self.predict_file)
        else:
            ori_test_file = os.path.join(self.data_dir, "thuc_news.test.txt")
            self.num_actual_predict_examples = len(processor._read_json(ori_test_file))



    def init_estimator(self):
        tpu_cluster_resolver = None  # tpu集群处理
        if self.use_tpu and self.tpu_name:
            tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
                self.tpu_name, zone=self.tpu_zone, project=self.gcp_project)

        is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2  # per_host：每主机，XLnet中num_core_per_host指的是每主机核数
        run_config = tf.contrib.tpu.RunConfig(  # tpu的运行配置
            cluster=tpu_cluster_resolver,
            master=self.master,
            model_dir=self.output_dir,
            save_checkpoints_steps=self.save_checkpoints_steps,
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=self.iterations_per_loop,  # 在每个estimator调用中执行多少步，default中为1000步
                num_shards=self.num_tpu_cores,  # # tpu核数，default为8
                per_host_input_for_training=is_per_host))

        model_fn = self.bert_model.model_fn_builder(
            init_checkpoint=self.init_checkpoint,
            learning_rate=self.learning_rate,
            num_train_steps=self.num_train_steps,
            num_warmup_steps=self.num_warmup_steps,
            use_tpu=self.use_tpu,
            use_one_hot_embeddings=self.use_tpu)

        # If TPU is not available, this will fall back to normal Estimator on CPU
        # or GPU.
        self.estimator = tf.contrib.tpu.TPUEstimator(
            use_tpu=self.use_tpu,
            model_fn=model_fn,
            config=run_config,
            train_batch_size=self.train_batch_size,
            eval_batch_size=self.eval_batch_size,
            predict_batch_size=self.predict_batch_size)


    def train(self, train_file, processor):

        self.log.info("***** Running training *****")
        self.log.info("  Num examples = %d", self.num_train_examples)
        self.log.info("  Batch size = %d", self.train_batch_size)
        self.log.info("  Num steps = %d", self.num_train_steps)
        # 调用此函数，完成：1.TFRecord to example 2.int64 to int32
        train_input_fn = processor.file_based_input_fn_builder(
            input_file=train_file,
            seq_length=self.max_seq_length,
            is_training=True,
            drop_remainder=True)
        self.estimator.train(input_fn=train_input_fn, max_steps=self.num_train_steps)


    def eval(self, eval_file, processor):
        self.log.info("***** Running evaluation *****")
        # self.log.info("  Num examples = %d (%d actual, %d padding)",
        #                 len(eval_examples), self.num_actual_eval_examples,
        #                 len(eval_examples) - num_actual_eval_examples)
        self.log.info("  Batch size = %d", self.eval_batch_size)

        # This tells the estimator to run through the entire set.
        eval_steps = None
        # However, if running eval on the TPU, you will need to specify the
        # number of steps.
        # if self.use_tpu:
        #     # 假定使用TPU时，前面整除处理已经成功，将得到eval_steps为整数值
        #     assert len(eval_examples) % self.eval_batch_size == 0
        #     eval_steps = int(len(eval_examples) // self.eval_batch_size)

        # 如果使用tpu的话，删除剩余的部分（可能是无法整除的部分）
        eval_drop_remainder = True if self.use_tpu else False
        eval_input_fn = processor.file_based_input_fn_builder(
            input_file=eval_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=eval_drop_remainder)

        result = self.estimator.evaluate(input_fn=eval_input_fn, steps=eval_steps)

        output_eval_file = os.path.join(self.output_dir, "eval_results.txt")
        with tf.gfile.GFile(output_eval_file, "w") as writer:
            self.log.info("***** Eval results *****")
            for key in sorted(result.keys()):
                self.log.info("  %s = %s", key, str(result[key]))
                writer.write("%s = %s\n" % (key, str(result[key])))


    def predict(self, predict_file, processor):
        self.log.info("***** Running prediction*****")
        # self.log.info("  Num examples = %d (%d actual, %d padding)",
        #                 len(predict_examples), self.num_actual_predict_examples,
        #                 len(predict_examples) - self.num_actual_predict_examples)
        self.log.info("  Batch size = %d", self.predict_batch_size)

        predict_drop_remainder = True if self.use_tpu else False
        predict_input_fn = processor.file_based_input_fn_builder(
            input_file=predict_file,
            seq_length=self.max_seq_length,
            is_training=False,
            drop_remainder=predict_drop_remainder)

        result = self.estimator.predict(input_fn=predict_input_fn)

        # 写测试集预测结果文件
        output_predict_file = os.path.join(self.output_dir, "test_results.tsv")
        with tf.gfile.GFile(output_predict_file, "w") as writer:
            num_written_lines = 0
            self.log.info("***** Predict results *****")
            for (i, prediction) in enumerate(result):
                # 写入probabilities的键值对，比如二分类：有预测为0的一列，预测为1的一列
                probabilities = prediction["probabilities"]
                if i >= self.num_actual_predict_examples:
                    break
                output_line = "\t".join(
                    str(class_probability)
                    for class_probability in probabilities) + "\n"
                writer.write(output_line)
                num_written_lines += 1
        assert num_written_lines == self.num_actual_predict_examples


def main():
    model_path = os.path.join(DATA_PATH, "model", "thuc_news")
    log_file = os.path.join(model_path, 'bert_train_log')
    log = Logger("bert_train_log", log2console=False, log2file=True, logfile=log_file).get_logger()
    conf_file = os.path.join(CONFIG_PATH, "bert_model_config.ini")
    trainer = ThucNewsTrainer(conf_file, logger=log)
    processor = trainer.processor
    train_file = trainer.train_file
    eval_file = trainer.eval_file
    test_file = trainer.predict_file
    trainer.train(train_file, processor)
    trainer.eval(eval_file, processor)
    trainer.predict(test_file, processor)


if __name__ == "__main__":
    main()