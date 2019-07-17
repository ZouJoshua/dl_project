#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-7-17 上午10:18
@File    : han_classifier.py
@Desc    : 分层注意网络
"""

from keras import Input, Model
from keras.layers import Embedding, Bidirectional, CuDNNLSTM, TimeDistributed, Dense, Reshape

from .attention_layer import Attention
from .basic_classifier import TextClassifier


class TextHanClassifier(TextClassifier):

    # 对长文本比较好, 可以在长文本中截断处理，把一段作为一个sentence
    def build_model(self):
        input_word = Input(shape=(int(self.maxlen / 5),))
        x_word = Embedding(len(self.embeddings),
                           300,
                           weights=[self.embeddings],
                           trainable=False)(input_word)
        x_word = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x_word)
        x_word = Attention()(x_word)
        model_word = Model(input_word, x_word)

        # Sentence part
        inputs = Input(shape=(self.maxlen,))  # (5,self.maxlen) 代表：(篇章最多包含的句子，每句包含的最大词数)
        reshape = Reshape((5, int(self.maxlen / 5)))(inputs)
        x_sentence = TimeDistributed(model_word)(reshape)
        x_sentence = Bidirectional(CuDNNLSTM(128, return_sequences=True))(x_sentence)
        x_sentence = Attention()(x_sentence)

        output = Dense(1, activation='sigmoid')(x_sentence)
        model = Model(inputs=inputs, outputs=output)
        model.compile('adam', 'binary_crossentropy', metrics=['accuracy'])
        return model

    def train(self, batch_size=512, epochs=20):
        # 比较耗费资源，笔记本GPU跑不动，只好减小batch_size
        return super(TextHanClassifier, self).train(128, 2)