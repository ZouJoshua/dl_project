#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/31/19 3:00 PM
@File    : test.py
@Desc    : 

"""

import numpy as np
import pandas as pd
import tensorflow as tf
from model_tensorflow.vgg_text_generator_model import TextGenerator
from nlp_tasks.text_generation.flickr30K.train import preProBuildWordVocab

###### Parameters ######
n_epochs = 1000
batch_size = 80
dim_embed = 256
dim_ctx = 512
dim_hidden = 256
ctx_shape = [196, 512]
pretrained_model_path = './model/model-8'
#############################
annotation_path = './data/annotations.pickle'
feat_path = './data/feats.npy'
model_path = './model/'
#############################


def test(test_feat='./guitar_player.npy', model_path='./model/model-6', maxlen=20):
    annotation_data = pd.read_pickle(annotation_path)
    captions = annotation_data['caption'].values
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)
    n_words = len(wordtoix)
    feat = np.load(test_feat).reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1, 2)

    sess = tf.InteractiveSession()

    caption_generator = TextGenerator(
            n_words=n_words,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen,
            batch_size=batch_size,
            ctx_shape=ctx_shape)

    context, generated_words, logit_list, alpha_list = caption_generator.build_generator(maxlen=maxlen)
    saver = tf.train.Saver()
    saver.restore(sess, model_path)

    generated_word_index = sess.run(generated_words, feed_dict={context:feat})
    alpha_list_val = sess.run(alpha_list, feed_dict={context:feat})
    generated_words = [ixtoword[x[0]] for x in generated_word_index]
    punctuation = np.argmax(np.array(generated_words) == '.')+1

    generated_words = generated_words[:punctuation]
    alpha_list_val = alpha_list_val[:punctuation]
    return generated_words, alpha_list_val

#    generated_sentence = ' '.join(generated_words)
#    ipdb.set_trace()