#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/31/19 3:00 PM
@File    : vgg_text_generator_model.py
@Desc    : 

"""


import math
import tensorflow as tf
import numpy as np



class TextGenerator(object):

    def init_weight(self, dim_in, dim_out, name=None, stddev=1.0):
        return tf.Variable(tf.truncated_normal([dim_in, dim_out], stddev=stddev/math.sqrt(float(dim_in))), name=name)

    def init_bias(self, dim_out, name=None):
        return tf.Variable(tf.zeros([dim_out]), name=name)

    def __init__(self, n_words, dim_embed, dim_ctx, dim_hidden, n_lstm_steps, batch_size=200, ctx_shape=[196, 512], bias_init_vector=None):
        self.n_words = n_words
        self.dim_embed = dim_embed
        self.dim_ctx = dim_ctx
        self.dim_hidden = dim_hidden
        self.ctx_shape = ctx_shape
        self.n_lstm_steps = n_lstm_steps
        self.batch_size = batch_size

        with tf.device("/cpu:0"):
            self.Wemb = tf.Variable(tf.random_uniform([n_words, dim_embed], -1.0, 1.0), name='Wemb')

        self.init_hidden_W = self.init_weight(dim_ctx, dim_hidden, name='init_hidden_W')
        self.init_hidden_b = self.init_bias(dim_hidden, name='init_hidden_b')

        self.init_memory_W = self.init_weight(dim_ctx, dim_hidden, name='init_memory_W')
        self.init_memory_b = self.init_bias(dim_hidden, name='init_memory_b')

        self.lstm_W = self.init_weight(dim_embed, dim_hidden*4, name='lstm_W')
        self.lstm_U = self.init_weight(dim_hidden, dim_hidden*4, name='lstm_U')
        self.lstm_b = self.init_bias(dim_hidden*4, name='lstm_b')

        self.image_encode_W = self.init_weight(dim_ctx, dim_hidden*4, name='image_encode_W')

        self.image_att_W = self.init_weight(dim_ctx, dim_ctx, name='image_att_W')
        self.hidden_att_W = self.init_weight(dim_hidden, dim_ctx, name='hidden_att_W')
        self.pre_att_b = self.init_bias(dim_ctx, name='pre_att_b')

        self.att_W = self.init_weight(dim_ctx, 1, name='att_W')
        self.att_b = self.init_bias(1, name='att_b')

        self.decode_lstm_W = self.init_weight(dim_hidden, dim_embed, name='decode_lstm_W')
        self.decode_lstm_b = self.init_bias(dim_embed, name='decode_lstm_b')

        self.decode_word_W = self.init_weight(dim_embed, n_words, name='decode_word_W')

        if bias_init_vector is not None:
            self.decode_word_b = tf.Variable(bias_init_vector.astype(np.float32), name='decode_word_b')
        else:
            self.decode_word_b = self.init_bias(n_words, name='decode_word_b')


    def get_initial_lstm(self, mean_context):
        initial_hidden = tf.nn.tanh(tf.matmul(mean_context, self.init_hidden_W) + self.init_hidden_b)
        initial_memory = tf.nn.tanh(tf.matmul(mean_context, self.init_memory_W) + self.init_memory_b)

        return initial_hidden, initial_memory

    def build_model(self):
        context = tf.placeholder("float32", [self.batch_size, self.ctx_shape[0], self.ctx_shape[1]])
        sentence = tf.placeholder("int32", [self.batch_size, self.n_lstm_steps])
        mask = tf.placeholder("float32", [self.batch_size, self.n_lstm_steps])

        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))

        # TensorFlow가 dot(3D tensor, matrix)
        context_flat = tf.reshape(context, [-1, self.dim_ctx])
        context_encode = tf.matmul(context_flat, self.image_att_W)  # (batch_size, 196, 512)
        context_encode = tf.reshape(context_encode, [-1, self.ctx_shape[0], self.ctx_shape[1]])

        loss = 0.0

        for ind in range(self.n_lstm_steps):

            if ind == 0:
                word_emb = tf.zeros([self.batch_size, self.dim_embed])
            else:
                tf.get_variable_scope().reuse_variables()
                with tf.device("/cpu:0"):
                    word_emb = tf.nn.embedding_lookup(self.Wemb, sentence[:,ind-1])

            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b  # (batch_size, hidden*4)

            labels = tf.expand_dims(sentence[:,ind], 1)
            indices = tf.expand_dims(tf.range(0, self.batch_size, 1), 1)
            concated = tf.concat(1, [indices, labels])
            onehot_labels = tf.sparse_to_dense( concated, tf.pack([self.batch_size, self.n_words]), 1.0, 0.0)

            context_encode = context_encode + \
                 tf.expand_dims(tf.matmul(h, self.hidden_att_W), 1) + \
                 self.pre_att_b

            context_encode = tf.nn.tanh(context_encode)

            # context_encode: 3D -> flat required
            context_encode_flat = tf.reshape(context_encode, [-1, self.dim_ctx])  # (batch_size*196, 512)
            alpha = tf.matmul(context_encode_flat, self.att_W) + self.att_b  # (batch_size*196, 1)
            alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]])
            alpha = tf.nn.softmax(alpha)

            weighted_context = tf.reduce_sum(context * tf.expand_dims(alpha, 2), 1)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)
            i, f, o, new_c = tf.split(1, 4, lstm_preactive)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f * c + i * new_c
            h = o * tf.nn.tanh(new_c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)
            logits = tf.nn.dropout(logits, 0.5)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logit_words, onehot_labels)
            cross_entropy = cross_entropy * mask[:,ind]

            current_loss = tf.reduce_sum(cross_entropy)
            loss = loss + current_loss

        loss = loss / tf.reduce_sum(mask)
        return loss, context, sentence, mask

    def build_generator(self, maxlen):
        context = tf.placeholder("float32", [1, self.ctx_shape[0], self.ctx_shape[1]])
        h, c = self.get_initial_lstm(tf.reduce_mean(context, 1))

        context_encode = tf.matmul(tf.squeeze(context), self.image_att_W)
        generated_words = []
        logit_list = []
        alpha_list = []
        word_emb = tf.zeros([1, self.dim_embed])
        for ind in range(maxlen):
            x_t = tf.matmul(word_emb, self.lstm_W) + self.lstm_b
            context_encode = context_encode + tf.matmul(h, self.hidden_att_W) + self.pre_att_b
            context_encode = tf.nn.tanh(context_encode)

            alpha = tf.matmul(context_encode, self.att_W) + self.att_b
            alpha = tf.reshape(alpha, [-1, self.ctx_shape[0]] )
            alpha = tf.nn.softmax(alpha)

            alpha = tf.reshape(alpha, (self.ctx_shape[0], -1))
            alpha_list.append(alpha)

            weighted_context = tf.reduce_sum(tf.squeeze(context) * alpha, 0)
            weighted_context = tf.expand_dims(weighted_context, 0)

            lstm_preactive = tf.matmul(h, self.lstm_U) + x_t + tf.matmul(weighted_context, self.image_encode_W)

            i, f, o, new_c = tf.split(1, 4, lstm_preactive)

            i = tf.nn.sigmoid(i)
            f = tf.nn.sigmoid(f)
            o = tf.nn.sigmoid(o)
            new_c = tf.nn.tanh(new_c)

            c = f*c + i*new_c
            h = o*tf.nn.tanh(new_c)

            logits = tf.matmul(h, self.decode_lstm_W) + self.decode_lstm_b
            logits = tf.nn.relu(logits)

            logit_words = tf.matmul(logits, self.decode_word_W) + self.decode_word_b

            max_prob_word = tf.argmax(logit_words, 1)

            with tf.device("/cpu:0"):
                word_emb = tf.nn.embedding_lookup(self.Wemb, max_prob_word)

            generated_words.append(max_prob_word)
            logit_list.append(logit_words)

        return context, generated_words, logit_list, alpha_list


