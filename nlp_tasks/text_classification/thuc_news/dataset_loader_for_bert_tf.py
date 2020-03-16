#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 3/14/20 8:47 AM
@File    : dataset_loader_for_bert_tf.py
@Desc    : 

"""


import os
import json
import random
from sklearn.utils import shuffle
import tqdm
import logging

from model_tensorflow.bert_model import tokenization


class InputFeatures(object):
  """A single set of features of data."""

  def __init__(self,
               input_ids,
               input_mask,
               segment_ids,
               label_id,
               is_real_example=True):
    self.input_ids = input_ids
    self.input_mask = input_mask
    self.segment_ids = segment_ids
    self.label_id = label_id
    self.is_real_example = is_real_example



class DatasetLoader(object):
    def __init__(self, config, logger=None):
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("bert_train_log")
            logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s')
            logging.root.setLevel(level=logging.INFO)

        self.vocab_file = config.get("vocab_file")
        self.output_dir = config.get("output_dir")
        self.label2idx_path = config.get("label2idx_path")
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        self._sequence_length = config.get("sequence_length")  # 每条输入的序列处理为定长
        self.label_map = self.label_to_index()

    @staticmethod
    def read_data(corpus_path, mode=None):
        """
        加载语料, 读取数据
        :param corpus_path:
        :param mode: train,eval,test
        :return:
        """

        with open(corpus_path, "r", encoding="utf-8") as f:
            # 将数据集全部加载到内存
            lines = [eval(line) for line in tqdm.tqdm(f, desc="Loading {} dataset".format(mode))]
            # 打乱顺序
            if mode == "train":
                lines = shuffle(lines)
            # 获取数据长度(条数)
            # corpus_lines = len(lines)
            return lines

    def _get_text_and_label(self, dict_line):
        # 获取文本和标记
        text = dict_line["text"]
        label = dict_line["label"]
        return text, label


    def convert_examples_to_features(self, lines, set_type):
        """
        将文本表达转化为索引表征
        :param lines:
        :param set_type: train,eval,test
        :return:
        """
        features = []
        tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_file, do_lower_case=True)
        for (i, line) in enumerate(lines):
            guid = "%s-%s" % (set_type, i)
            text, label = self._get_text_and_label(line)
            feature = self.convert_single_example_to_feature(guid, tokenizer, text, label=label)
            features.append(feature)

        return features


    def convert_single_example_to_feature(self, guid, tokenizer, text, label=None):
        text = tokenization.convert_to_unicode(text)
        if label:
            label = tokenization.convert_to_unicode(label)
            label_id = self.label_map.get(label)
        else:
            label_id = None

        tokens = tokenizer.tokenize(text)
        tokens = ["[CLS]"] + tokens + ["[SEP]"]
        input_id = tokenizer.convert_tokens_to_ids(tokens)
        input_mask = [1] * len(input_id)
        segment_id = [0] * len(input_id)

        if guid.split("-")[1] < 5:
            self.log.info("*** Example ***")
            self.log.info("guid: %s" % (guid))
            self.log.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in tokens]))
            self.log.info("input_ids: %s" % " ".join([str(x) for x in input_id]))
            self.log.info("input_mask: %s" % " ".join([str(x) for x in input_mask]))
            self.log.info("segment_ids: %s" % " ".join([str(x) for x in segment_id]))
            self.log.info("label: %s (id = %d)" % (label, label_id))

        pad_input_id, pad_input_mask, pad_segment_id = self.zero_padding(input_id, input_mask, segment_id)


        feature = InputFeatures(
            input_ids=pad_input_id,
            input_mask=pad_input_mask,
            segment_ids=pad_segment_id,
            label_id=label_id,
            is_real_example=True)

        return feature


    def label_to_index(self):
        if os.path.exists(self.label2idx_path):
            with open(self.label2idx_path, "r", encoding="utf-8") as fr:
                return json.load(fr)
        else:
            raise FileNotFoundError


    def zero_padding(self, input_id, input_mask, segment_id):
        """
        对序列进行补全
        :param input_id:
        :param input_mask:
        :param segment_id:
        :return:
        """

        if len(input_id) < self._sequence_length:
            pad_input_id = input_id + [0] * (self._sequence_length - len(input_id))
            pad_input_mask = input_mask + [0] * (self._sequence_length - len(input_mask))
            pad_segment_id = segment_id + [0] * (self._sequence_length - len(segment_id))
        else:
            pad_input_id = input_id[:self._sequence_length]
            pad_input_mask = input_mask[:self._sequence_length]
            pad_segment_id = segment_id[:self._sequence_length]

        return pad_input_id, pad_input_mask, pad_segment_id

    def gen_data(self, file_path, mode):
        """
        :param file_path:
        :param mode: train,eval,test
        :return:
        """
        lines = self.read_data(file_path)
        features = self.convert_examples_to_features(lines, set_type=mode)

        return features

    def next_batch(self, features, batch_size, mode="train"):
        """
        生成batch数据
        :param input_ids:
        :param input_masks:
        :param segment_ids:
        :param label_ids:
        :return:
        """

        # random.shuffle(features)
        if mode == "train":
            shuffle(features)

        num_batches = len(features) // batch_size

        for i in range(num_batches):
            start = i * batch_size
            end = start + batch_size
            batch_features = features[start: end]
            batch_input_ids = [batch.input_ids for batch in batch_features]
            batch_input_masks = [batch.input_mask for batch in batch_features]
            batch_segment_ids = [batch.segment_ids for batch in batch_features]
            batch_label_ids = [batch.label_id for batch in batch_features]

            yield dict(input_ids=batch_input_ids,
                       input_masks=batch_input_masks,
                       segment_ids=batch_segment_ids,
                       label_ids=batch_label_ids)