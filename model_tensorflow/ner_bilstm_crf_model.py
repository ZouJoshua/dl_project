#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/12/19 8:06 PM
@File    : ner_bilstm_crf_model.py
@Desc    : 

"""
import tensorflow as tf
import numpy as np


def data_type():
    return tf.float32


class NERTagger(object):
    """The NER Tagger Model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.is_training = is_training
        self.crf_layer = config.crf_layer  # if the model has the final CRF decoding layer
        size = config.hidden_size
        vocab_size = config.vocab_size

        # Define input and target tensors
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        # BiLSTM CRF model
        self._cost, self._logits, self._transition_params = self._bilstm_crf_model(inputs, self._targets, config)

        # Gradients and SGD update operation for training the model.
        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self._cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(
            zip(grads, tvars),
            global_step=tf.contrib.framework.get_or_create_global_step())

        self._new_lr = tf.placeholder(data_type(), shape=[], name="new_learning_rate")
        self._lr_update = tf.assign(self._lr, self._new_lr)
        self.saver = tf.train.Saver(tf.global_variables())

    def assign_lr(self, session, lr_value):
        session.run(self._lr_update, feed_dict={self._new_lr: lr_value})

    @property
    def input_data(self):
        return self._input_data

    @property
    def targets(self):
        return self._targets

    @property
    def cost(self):
        return self._cost

    @property
    def logits(self):
        return self._logits

    @property
    def transition_params(self):  # transition params for CRF layer
        return self._transition_params

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def accuracy(self):
        return self._accuracy

    def _bilstm_crf_model(self, inputs, targets, config):
        '''
        @Use BasicLSTMCell, MultiRNNCell method to build LSTM model
        @Use CRF layer to calculate log likelihood and viterbi decoder to caculate the optimal sequence
        @return logits, cost and others
        '''
        batch_size = config.batch_size
        num_steps = config.num_steps
        num_layers = config.num_layers
        size = config.hidden_size
        vocab_size = config.vocab_size
        target_num = config.target_num  # target output number
        num_features = 2 * size

        # Bi-LSTM NN layer
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])

        initial_state_fw = cell_fw.zero_state(batch_size, data_type())
        initial_state_bw = cell_bw.zero_state(batch_size, data_type())

        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        inputs_list = [tf.squeeze(s, axis=1) for s in tf.split(value=inputs, num_or_size_splits=num_steps, axis=1)]

        with tf.variable_scope("pos_bilstm_crf"):
            outputs, state_fw, state_bw = tf.nn.static_bidirectional_rnn(
                cell_fw, cell_bw, inputs_list, initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw)

        # outputs is a length T list of output vectors, which is [batch_size, 2 * hidden_size]
        # [time][batch][cell_fw.output_size + cell_bw.output_size]

        output = tf.reshape(tf.concat(outputs, 1), [-1, num_features])
        # output has size: batch_size, [T, size * 2]
        print("LSTM NN layer output size:")
        print(output.get_shape())

        # Linear-Chain CRF Layer
        x_crf_input = tf.reshape(output, [batch_size, num_steps, num_features])
        crf_weights = tf.get_variable("crf_weights", [num_features, target_num], dtype=data_type())
        matricized_crf_input = tf.reshape(x_crf_input, [-1, num_features])
        matricized_unary_scores = tf.matmul(matricized_crf_input, crf_weights)
        unary_scores = tf.reshape(matricized_unary_scores, [batch_size, num_steps, target_num])

        # log-likelihood
        sequence_lengths = tf.constant(np.full(batch_size, num_steps - 1, dtype=np.int32))  # shape: [batch_size], value: [T-1, T-1,...]
        log_likelihood, transition_params = tf.contrib.crf.crf_log_likelihood(unary_scores,
                                                                              targets, sequence_lengths)

        # Add a training op to tune the parameters.
        loss = tf.reduce_mean(-log_likelihood)
        logits = unary_scores  # CRF x input, shape [batch_size, num_steps, target_num]
        cost = loss
        return cost, logits, transition_params