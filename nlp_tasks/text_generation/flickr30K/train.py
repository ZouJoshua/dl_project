#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/31/19 2:59 PM
@File    : train.py
@Desc    : 

"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from model_tensorflow.vgg_text_generator_model import TextGenerator
from keras.preprocessing import sequence



def preProBuildWordVocab(sentence_iterator, word_count_threshold=30): # borrowed this function from NeuralTalk
    print('preprocessing word counts and creating vocab based on word count threshold %d' % (word_count_threshold, ))
    word_counts = {}
    nsents = 0
    for sent in sentence_iterator:
      nsents += 1
      for w in sent.lower().split(' '):
        word_counts[w] = word_counts.get(w, 0) + 1
    vocab = [w for w in word_counts if word_counts[w] >= word_count_threshold]
    print('filtered words from %d to %d' % (len(word_counts), len(vocab)))

    ixtoword = {}
    ixtoword[0] = '.'  # period at the end of the sentence. make first dimension be end token
    wordtoix = {}
    wordtoix['#START#'] = 0 # make first vector be the start token
    ix = 1
    for w in vocab:
      wordtoix[w] = ix
      ixtoword[ix] = w
      ix += 1

    word_counts['.'] = nsents
    bias_init_vector = np.array([1.0*word_counts[ixtoword[i]] for i in ixtoword])
    bias_init_vector /= np.sum(bias_init_vector) # normalize to frequencies
    bias_init_vector = np.log(bias_init_vector)
    bias_init_vector -= np.max(bias_init_vector) # shift to nice numeric range
    return wordtoix, ixtoword, bias_init_vector


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


def train(pretrained_model_path=pretrained_model_path):
    annotation_data = pd.read_pickle(annotation_path)
    captions = annotation_data['caption'].values
    wordtoix, ixtoword, bias_init_vector = preProBuildWordVocab(captions)

    learning_rate=0.001
    n_words = len(wordtoix)
    feats = np.load(feat_path)
    maxlen = np.max(map(lambda x: len(x.split(' ')), captions))

    sess = tf.InteractiveSession()

    caption_generator = TextGenerator(
            n_words=n_words,
            dim_embed=dim_embed,
            dim_ctx=dim_ctx,
            dim_hidden=dim_hidden,
            n_lstm_steps=maxlen+1,  # w1~wN预测到最后一次
            batch_size=batch_size,
            ctx_shape=ctx_shape,
            bias_init_vector=bias_init_vector)

    loss, context, sentence, mask = caption_generator.build_model()
    saver = tf.train.Saver(max_to_keep=50)

    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    tf.initialize_all_variables().run()
    if pretrained_model_path is not None:
        print("Starting with pretrained model")
        saver.restore(sess, pretrained_model_path)

    index = list(annotation_data.index)
    np.random.shuffle(index)
    annotation_data = annotation_data.ix[index]

    captions = annotation_data['caption'].values
    image_id = annotation_data['image_id'].values

    for epoch in range(n_epochs):
        for start, end in zip(range(0, len(captions), batch_size),
                range(batch_size, len(captions), batch_size)):

            current_feats = feats[image_id[start:end]]
            current_feats = current_feats.reshape(-1, ctx_shape[1], ctx_shape[0]).swapaxes(1,2)

            current_captions = captions[start:end]
            current_caption_ind = map(lambda cap: [wordtoix[word] for word in cap.lower().split(' ')[:-1] if word in wordtoix], current_captions) # '.'은 제거

            current_caption_matrix = sequence.pad_sequences(current_caption_ind, padding='post', maxlen=maxlen+1)

            current_mask_matrix = np.zeros((current_caption_matrix.shape[0], current_caption_matrix.shape[1]))
            nonzeros = np.array( map(lambda x: (x != 0).sum()+1, current_caption_matrix ))

            for ind, row in enumerate(current_mask_matrix):
                row[:nonzeros[ind]] = 1

            _, loss_value = sess.run([train_op, loss], feed_dict={
                context:current_feats,
                sentence:current_caption_matrix,
                mask:current_mask_matrix})

            print("Current Cost: ", loss_value)
        saver.save(sess, os.path.join(model_path, 'model'), global_step=epoch)

