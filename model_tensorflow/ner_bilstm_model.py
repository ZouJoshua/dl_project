#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/12/19 6:07 PM
@File    : ner_bilstm_model.py
@Desc    : 

"""

import tensorflow as tf


def data_type():
    return tf.float32

class NERTagger(object):
    """The NER Tagger Model."""

    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.is_training = is_training
        size = config.hidden_size
        vocab_size = config.vocab_size

        # Define input and target tensors
        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

        with tf.device("/cpu:0"):
            embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
            inputs = tf.nn.embedding_lookup(embedding, self._input_data)

        if (config.bi_direction):  # BiLSTM
            self._cost, self._logits = self._bilstm_model(inputs, self._targets, config)
        else:  # LSTM
            self._cost, self._logits, self._final_state, self._initial_state = self._lstm_model(inputs, self._targets, config)

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
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

    @property
    def accuracy(self):
        return self._accuracy



    def _lstm_model(self, inputs, targets, config):
        '''
        @Use BasicLSTMCell and MultiRNNCell class to build LSTM model,
        @return logits, cost and others
        '''
        batch_size = config.batch_size
        num_steps = config.num_steps
        num_layers = config.num_layers
        size = config.hidden_size
        vocab_size = config.vocab_size
        target_num = config.target_num  # target output number

        # multi-layer LSTM cells
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])
        initial_state = cell.zero_state(batch_size, data_type())

        outputs = []  # outputs shape: list of tensor with shape [batch_size, size], length: time_step
        state = initial_state
        with tf.variable_scope("ner_lstm"):
            for time_step in range(num_steps):
                if time_step > 0: tf.get_variable_scope().reuse_variables()
                (cell_output, state) = cell(inputs[:, time_step, :], state)  # inputs[batch_size, time_step, hidden_size]
                outputs.append(cell_output)

        output = tf.reshape(tf.concat(outputs, 1), [-1, size])  # output shape [time_step, size]
        softmax_w = tf.get_variable("softmax_w", [size, target_num], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b

        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]), logits=logits)
        cost = tf.reduce_sum(loss) / batch_size  # loss [time_step]

        # adding extra statistics to monitor
        correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(targets, [-1]))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return cost, logits, state, initial_state


    def _bilstm_model(self, inputs, targets, config):
        '''
        @Use BasicLSTMCell, MultiRNNCell method to build LSTM model,
        @return logits, cost and others
        '''
        batch_size = config.batch_size
        num_steps = config.num_steps
        num_layers = config.num_layers
        size = config.hidden_size
        vocab_size = config.vocab_size
        target_num = config.target_num  # target output number

        # NOTICE: Changes in TF 1.2, create LSTM layer with different variables
        # Multi-Layer Forward LSTM Cell
        cell_fw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])
        # Multi-Layer Backward LSTM Cell
        cell_bw = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(num_layers)])

        initial_state_fw = cell_fw.zero_state(batch_size, data_type())
        initial_state_bw = cell_bw.zero_state(batch_size, data_type())

        # Split to get a list of 'n_steps' tensors of shape (batch_size, n_input)
        inputs_list = [tf.squeeze(s, axis=1) for s in tf.split(value=inputs, num_or_size_splits=num_steps, axis=1)]

        with tf.variable_scope("ner_bilstm"):
            outputs, state_fw, state_bw = tf.nn.static_bidirectional_rnn(
                cell_fw, cell_bw, inputs_list, initial_state_fw=initial_state_fw,
                initial_state_bw=initial_state_bw)

        # outputs is a length T list of output vectors, which is [batch_size, 2 * hidden_size]
        # [time][batch][cell_fw.output_size + cell_bw.output_size]

        output = tf.reshape(tf.concat(outputs, 1), [-1, size * 2])
        # output has size: [T, size * 2]

        softmax_w = tf.get_variable("softmax_w", [size * 2, target_num], dtype=data_type())
        softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
        logits = tf.matmul(output, softmax_w) + softmax_b

        correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(targets, [-1]))
        self._accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(targets, [-1]), logits=logits)
        cost = tf.reduce_sum(loss) / batch_size  # loss [time_step]
        return cost, logits