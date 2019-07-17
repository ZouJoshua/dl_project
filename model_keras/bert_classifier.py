#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-17 上午10:16
@File    : bert_classifier.py
@Desc    : 
"""


from keras import Input, Model
from keras.layers import Lambda, Dense
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import codecs
from sklearn.model_selection import train_test_split


class ZhTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


class BertTextClassifier:
    """
    目前示例只是拿谷歌训练好的模型来使用，不涉及训练过程（训练成本太高）
    """

    def __init__(self, config_path,
                 checkpoint_path,
                 dict_path,
                 train=False,
                 data_path=None):
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.dict_path = dict_path

        if not train:
            pass
        else:
            self.data_path = data_path
            self.model = self.train()

    def train(self):
        x1_train, x2_train, y_train, x1_test, x2_test, y_test = self.preprocess()
        model = self.build_model()
        model.fit(
            [x1_train, x2_train], y_train,
            batch_size=8,
            epochs=5,
            validation_data=([x1_test, x2_test], y_test),
            verbose=1
        )
        return model

    def build_model(self):
        bert_model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path)

        for l in bert_model.layers:
            l.trainable = True

        x1_in = Input(shape=(None,))
        x2_in = Input(shape=(None,))

        x = bert_model([x1_in, x2_in])
        x = Lambda(lambda x: x[:, 0])(x)  # 取出[CLS]对应的向量用来做分类
        p = Dense(1, activation='sigmoid')(x)

        model = Model([x1_in, x2_in], p)
        model.compile(
            loss='binary_crossentropy',
            optimizer=Adam(1e-5),  # 用足够小的学习率
            metrics=['accuracy'])
        model.summary()
        return model

    def preprocess(self):
        token_dict = {}
        with codecs.open(self.dict_path, 'r', 'utf8') as reader:
            for line in reader:
                token = line.strip()
                token_dict[token] = len(token_dict)
        tokenizer = ZhTokenizer(token_dict)

        x_data, y_data = [], []
        with open(self.data_path, 'r', encoding='utf-8') as file:
            lines = file.readlines()
        for line in lines:
            line_ = line.split('##')
            x_data.append(line_[0])
            y_data.append(int(line_[1].strip()))

        x1, x2 = [], []

        for text in x_data:
            x1_, x2_ = tokenizer.encode(first=text)
            x1.append(x1_)
            x2.append(x2_)

        x1 = pad_sequences(x1, maxlen=100)
        x2 = pad_sequences(x2, maxlen=100)

        x1_train, x1_test, x2_train, x2_test, y_train, y_test = train_test_split(x1, x2, y_data)
        return x1_train, x2_train, y_train, x1_test, x2_test, y_test