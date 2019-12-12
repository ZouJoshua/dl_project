#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 12/11/19 12:15 AM
@File    : ner_with_bilstm_crf.py
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


from nlp_tasks.sequence_labeling import reader  # absolute import
from model_tensorflow.ner_bilstm_crf_model import NERTagger


# language option python command line: python ner_model_bilstm.py en
lang = "zh" if len(sys.argv) == 1 else sys.argv[1]  # default zh
file_path = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(file_path, "data", lang)  # path to find corpus vocab file
train_dir = os.path.join(file_path, "ckpt", lang)  # path to find model saved checkpoint file

flags = tf.flags
logging = tf.logging

flags.DEFINE_string("ner_lang", lang, "ner language option for model config")
flags.DEFINE_string("ner_data_path", data_path, "data_path")
flags.DEFINE_string("ner_train_dir", train_dir, "Training directory.")
flags.DEFINE_string("ner_scope_name", "ner_var_scope", "Define NER Tagging Variable Scope Name")

FLAGS = flags.FLAGS





# NER Model Configuration, Set Target Num, and input vocab_Size
class LargeConfigChinese(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 30
    hidden_size = 128
    max_epoch = 15
    max_max_epoch = 20
    keep_prob = 1.00
    lr_decay = 1 / 1.15
    batch_size = 1  # single sample batch
    vocab_size = 60000
    target_num = 8  # 7 NER Tags: nt, n, p, o, q (special), nz(entity_name), nbz(brand)
    bi_direction = True
    crf_layer = True


class LargeConfigEnglish(object):
    """Large config."""
    init_scale = 0.04
    learning_rate = 1.0
    max_grad_norm = 10
    num_layers = 2
    num_steps = 30
    hidden_size = 128
    max_epoch = 15
    max_max_epoch = 20
    keep_prob = 1.00
    lr_decay = 1 / 1.15
    batch_size = 1  # single sample batch
    vocab_size = 52000
    target_num = 15  # NER Tag 17, n, nf, nc, ne, (name, start, continue, end) n, p, o, q (special), nz entity_name, nbz
    bi_direction = True
    crf_layer = True


def get_config(lang):
    if (lang == 'zh'):
        return LargeConfigChinese()
    elif (lang == 'en'):
        return LargeConfigEnglish()
    # other lang options

    else:
        return None




def run_epoch(session, model, word_data, tag_data, eval_op, verbose=False):
    """Runs the model on the given data."""
    epoch_size = ((len(word_data) // model.batch_size) - 1) // model.num_steps

    start_time = time.time()
    costs = 0.0
    iters = 0
    correct_labels = 0  # prediction accuracy
    total_labels = 0

    for step, (x, y) in enumerate(reader.iterator(word_data, tag_data, model.batch_size, model.num_steps)):
        feed_dict = {}
        feed_dict[model.input_data] = x
        feed_dict[model.targets] = y
        fetches = []

        if (model.crf_layer):  # model has the CRF decoding layer
            fetches = [model.cost, model.logits, model.transition_params, eval_op]
            cost, logits, transition_params, _ = session.run(fetches, feed_dict)
            # iterate over batches [batch_size, num_steps, target_num], [batch_size, target_num]
            for unary_score_, y_ in zip(logits, y):  # unary_score_  :[num_steps, target_num], y_: [num_steps]
                viterbi_prediction = tf.contrib.crf.viterbi_decode(unary_score_, transition_params)
                # viterbi_prediction: tuple (list[id], value)
                # y_: tuple
                correct_labels += np.sum(np.equal(viterbi_prediction[0], y_))  # compare prediction sequence with golden sequence
                total_labels += len(y_)
                # print ("step %d:" % step)
                # print ("correct_labels %d" % correct_labels)
                # print ("viterbi_prediction")
                # print (viterbi_prediction)
        else:
            fetches = [model.cost, model.logits, eval_op]
            cost, logits, _ = session.run(fetches, feed_dict)

        costs += cost
        iters += model.num_steps

        if verbose and step % (epoch_size // 10) == 10:
            print("%.3f perplexity: %.3f speed: %.0f wps" %
                  (step * 1.0 / epoch_size, np.exp(costs / iters),
                   iters * model.batch_size / (time.time() - start_time)))

        # Accuracy
        if verbose and step % (epoch_size // 10) == 20:
            accuracy = 100.0 * correct_labels / float(total_labels)
            print("Accuracy: %.2f%%" % accuracy)

        # Save Model to CheckPoint when is_training is True
        if model.is_training:
            if step % (epoch_size // 10) == 10:
                checkpoint_path = os.path.join(FLAGS.ner_train_dir, "ner_bilstm_crf.ckpt")
                model.saver.save(session, checkpoint_path)
                print("Model Saved... at time step " + str(step))

    return np.exp(costs / iters)


def main(_):
    if not FLAGS.ner_data_path:
        raise ValueError("No data files found in 'data_path' folder")

    raw_data = reader.load_data(FLAGS.ner_data_path)
    train_word, train_tag, dev_word, dev_tag, test_word, test_tag, vocabulary = raw_data

    config = get_config(FLAGS.ner_lang)
    eval_config = get_config(FLAGS.ner_lang)
    eval_config.batch_size = 1
    eval_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:
        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope(FLAGS.ner_scope_name, reuse=None, initializer=initializer):
            m = NERTagger(is_training=True, config=config)
        with tf.variable_scope(FLAGS.ner_scope_name, reuse=True, initializer=initializer):
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