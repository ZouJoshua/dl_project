#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/12/19 5:17 PM
@File    : ner_lstm_model.py
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
        size = config.hidden_size
        vocab_size = config.vocab_size
        target_num = config.target_num  # target output number

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps], name="input_data")
        self._targets = tf.placeholder(tf.int32, [batch_size, num_steps], name="targets")

        # Check if Model is Training
        self.is_training = is_training

        # NOTICE: TF1.2 change API to make RNNcell share the same variables under namespace
        # Create multi-layer LSTM model, Separate Layers with different variables, we need to create multiple RNNCells separately
        cell = tf.nn.rnn_cell.MultiRNNCell([tf.nn.rnn_cell.BasicLSTMCell(size) for _ in range(config.num_layers)])
        self._initial_state = cell.zero_state(batch_size, data_type())

        with tf.variable_scope("ner_variables", reuse=tf.AUTO_REUSE):
            with tf.device("/cpu:0"):
                embedding = tf.get_variable("embedding", [vocab_size, size], dtype=data_type())
                inputs = tf.nn.embedding_lookup(embedding, self._input_data)

            if is_training and config.keep_prob < 1:
                inputs = tf.nn.dropout(inputs, config.keep_prob)

            outputs = []
            state = self._initial_state
            with tf.variable_scope("lstm", reuse=tf.AUTO_REUSE):
                for time_step in range(num_steps):
                    if time_step > 0: tf.get_variable_scope().reuse_variables()
                    (cell_output, state) = cell(inputs[:, time_step, :], state)
                    outputs.append(cell_output)

            output = tf.reshape(tf.concat(outputs, 1), [-1, size])
            softmax_w = tf.get_variable("softmax_w", [size, target_num], dtype=data_type())
            softmax_b = tf.get_variable("softmax_b", [target_num], dtype=data_type())
            # logits = tf.matmul(output, softmax_w) + softmax_b
            logits = tf.add(tf.matmul(output, softmax_w), softmax_b)
            prediction = tf.cast(tf.argmax(logits, 1), tf.int32, name="output_node")  # rename prediction to output_node for future inference

            loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tf.reshape(self._targets, [-1]), logits=logits)
            cost = tf.reduce_sum(loss) / batch_size  # loss [time_step]

            # adding extra statistics to monitor
            correct_prediction = tf.equal(tf.cast(tf.argmax(logits, 1), tf.int32), tf.reshape(self._targets, [-1]))
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

        # Fetch Reults in session.run()
        self._cost = cost = tf.reduce_sum(loss) / batch_size
        self._final_state = state
        self._logits = logits
        self._correct_prediction = correct_prediction

        self._lr = tf.Variable(0.0, trainable=False)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        optimizer = tf.train.GradientDescentOptimizer(self._lr)
        self._train_op = optimizer.apply_gradients(zip(grads, tvars))

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
    def initial_state(self):
        return self._initial_state

    @property
    def cost(self):
        return self._cost

    @property
    def final_state(self):
        return self._final_state

    @property
    def logits(self):
        return self._logits

    @property
    def correct_prediction(self):
        return self._correct_prediction

    @property
    def lr(self):
        return self._lr

    @property
    def train_op(self):
        return self._train_op

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

        # Multiple LSTM Cells
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
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        return cost, logits, state, initial_state