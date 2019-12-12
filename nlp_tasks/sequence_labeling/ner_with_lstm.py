#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/11/19 12:09 AM
@File    : ner_with_lstm.py
@Desc    : 

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals  # compatible with python3 unicode coding

import time
import numpy as np
import tensorflow as tf
import sys, os

from nlp_tasks.sequence_labeling import reader  # explicit relative import
from model_tensorflow.ner_lstm_model import NERTagger
from nlp_tasks.sequence_labeling.ner_model_util import get_model_var_scope
from nlp_tasks.sequence_labeling.ner_model_util import get_config, load_config
from nlp_tasks.sequence_labeling.ner_model_util import _ner_scope_name
from nlp_tasks.sequence_labeling.ner_model_util import _ner_variables_namescope

# language option python command line 'python ner_model.py zh'
lang = "zh" if len(sys.argv) == 1 else sys.argv[1]  # default zh
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "data", lang)
train_dir = os.path.join(file_path, "ckpt", lang)
modle_config_path = os.path.join(file_path, "data", "models.conf")

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("ner_lang", lang, "ner language option for model config")
flags.DEFINE_string("ner_data_path", data_path, "data_path")
flags.DEFINE_string("ner_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("ner_scope_name", _ner_scope_name, "Variable scope of NER Model")
flags.DEFINE_string("ner_model_config_path", modle_config_path, "Model hyper parameters configuration path")

FLAGS = flags.FLAGS




def run_epoch(session, model, word_data, tag_data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps
    start_time = time.time()
    costs = 0.0
    iters = 0
    correct_labels = 0
    total_labels = 0

    state = session.run(model.initial_state)
    for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size,
                                                  model.num_steps)):
        fetches = [model.cost, model.final_state, model.correct_prediction, eval_op]
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        for i, (c, h) in enumerate(model.initial_state):
            feed_dict[c] = state[i].c
            feed_dict[h] = state[i].h
        cost, state, correct_prediction, _ = session.run(fetches, feed_dict)
        costs += cost
        iters += model.num_steps

        correct_labels += np.sum(correct_prediction)
        total_labels += len(correct_prediction)

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

        if verbose and step % (epoch_size // 10) == 10:
            accuracy = 100.0 * correct_labels / float(total_labels)
            print("Cum Accuracy: %.2f%%" % accuracy)

        # Save Model to CheckPoint when is_training is True
        if model.is_training:
            if step % (epoch_size // 10) == 10:
                checkpoint_path = os.path.join(FLAGS.ner_train_dir, "ner.ckpt")
                model.saver.save(session, checkpoint_path)
                print("Model Saved... at time step " + str(step))

    return np.exp(costs / iters)


def main(_):
    if not FLAGS.ner_data_path:
        raise ValueError("No data files found in 'data_path' folder")

    # Load Data
    raw_data = reader.load_data(FLAGS.ner_data_path)
    train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocabulary = raw_data

    # Load Config
    config_dict = load_config(FLAGS.ner_model_config_path)
    config = get_config(config_dict, FLAGS.ner_lang)
    eval_config = get_config(config_dict, FLAGS.ner_lang)
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    # Load Model Variable Scope
    model_var_scope = get_model_var_scope(FLAGS.ner_scope_name, FLAGS.ner_lang)

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope(model_var_scope, reuse=True, initializer=initializer):
            m = NERTagger(is_training=True, config=config)
        with tf.variable_scope(model_var_scope, reuse=True, initializer=initializer):
            mvalid = NERTagger(is_training=False, config=config)
            mtest = NERTagger(is_training=False, config=eval_config)

        # CheckPoint State
        ckpt = tf.train.get_checkpoint_state(FLAGS.ner_train_dir)
        if ckpt:
            print("Loading model parameters from %s" % ckpt.model_checkpoint_path)
            m.saver.restore(session, tf.train.latest_checkpoint(FLAGS.ner_train_dir))
        else:
            print("Created model with fresh parameters.")
            session.run(tf.global_variables_initializer())

        # write the graph out for further use e.g. C++ API call
        tf.train.write_graph(session.graph_def, './models/', 'ner_graph.pbtxt', as_text=True)  # output is text

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(session, m, train_word, train_tag, m.train_op,
                                         verbose=True)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            valid_perplexity = run_epoch(session, mvalid, dev_word, dev_tag, tf.no_op())
            print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))

        test_perplexity = run_epoch(session, mtest, test_word, test_tag, tf.no_op())
        print("Test Perplexity: %.3f" % test_perplexity)


if __name__ == "__main__":
    tf.app.run()