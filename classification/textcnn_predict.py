#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:32
@File    : textcnn_predict.py
@Desc    : 

"""


import tensorflow as tf
import numpy as np
from preprocess.util import create_label_vocabulary, create_vocabulary, load_final_test_data, load_data_predict
from tflearn.data_utils import pad_sequences
import os
import codecs
from tf_model.textcnn_model import TextCNN

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("num_classes", 15, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 1, "Batch size for evaluating.")
tf.app.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.app.flags.DEFINE_string("ckpt_dir", "data/textcnn/textcnn_title_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("title_len", 200, "max sentence length")
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", False, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 15, "number of epochs.")
tf.app.flags.DEFINE_integer("validate_every", 1, "Validate every validate_every epochs.")
tf.app.flags.DEFINE_string("predict_target_file", "data/textcnn/textcnn_title_checkpoint/predict_target", "target file path for final prediction")
tf.app.flags.DEFINE_string("predict_source_file", 'data/textcnn/textcnn_title_checkpoint/predict_source', "target file path for final prediction")
tf.app.flags.DEFINE_integer("num_filters", 256, "number of filters")  # 128
tf.app.flags.DEFINE_string("ckpt_dir2", "text_cnn_title_desc_checkpoint_exp/", "checkpoint location for the model")

##############################################################################################################################################
filter_sizes = [3, 4, 5]  # todo: 找最佳的filter size
title_word_vector_path = "data/news_content/preprocess/title_word_vector.bin"
model_path = "data/news_content/preprocess/word2vec_model"

# 1.load data with vocabulary of words and labels
vocabulary_word2idx, vocabulary_idx2word = create_vocabulary(model_path, name_scope="cnn2")
vocab_size = len(vocabulary_word2idx)
vocabulary_word2index_label, vocabulary_index2word_label = create_label_vocabulary(name_scope="cnn2")
questionid_question_lists = load_final_test_data(FLAGS.predict_source_file)
test = load_data_predict(vocabulary_word2idx, vocabulary_word2index_label, questionid_question_lists)
testX = []
question_id_list = []
for tuple in test:
    question_id, question_string_list = tuple
    question_id_list.append(question_id)
    testX.append(question_string_list)
# 2.Data preprocessing: Sequence padding
print("start padding....")
testX2 = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
print("end padding...")
# 3.create session.
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
graph = tf.Graph().as_default()
global sess
global textCNN
with graph:
    sess = tf.Session(config=config)
    # 4.Instantiate Model
    textCNN = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size,
                  FLAGS.decay_steps, FLAGS.decay_rate,
                  FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)
    saver = tf.train.Saver()
    if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
        print("Restoring Variables from Checkpoint")
        saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
    else:
        print("Can't find the checkpoint.going to stop")
    # return
# 5.feed data, to get logits
number_of_training_data = len(testX2)
print("number_of_training_data:", number_of_training_data)
# index = 0
# predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
#############################################################################################################################################


def get_logits_with_value_by_input(start, end):
    x = testX2[start:end]
    global sess
    global textCNN
    logits = sess.run(textCNN.logits, feed_dict={textCNN.input_x: x, textCNN.dropout_keep_prob: 1})
    predicted_labels, value_labels = get_label_using_logits_with_value(logits[0], vocabulary_index2word_label)
    value_labels_exp = np.exp(value_labels)
    p_labels = value_labels_exp/np.sum(value_labels_exp)
    return predicted_labels, p_labels


def main(_):
    # 1.load data with vocabulary of words and labels
    vocabulary_word2index, vocabulary_index2word = create_vocabulary(simple='simple', word2vec_model_path=FLAGS.word2vec_model_path, name_scope="cnn2")
    vocab_size = len(vocabulary_word2index)
    vocabulary_word2index_label, vocabulary_index2word_label = create_label_vocabulary(name_scope="cnn2")
    questionid_question_lists=load_final_test_data(FLAGS.predict_source_file)
    test = load_data_predict(vocabulary_word2index,vocabulary_word2index_label,questionid_question_lists)
    testX =[]
    question_id_list = []
    for tuple in test:
        question_id, question_string_list = tuple
        question_id_list.append(question_id)
        testX.append(question_string_list)
    # 2.Data preprocessing: Sequence padding
    print("start padding....")
    testX2 = pad_sequences(testX, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("end padding...")
    # 3.create session.
    config = tf.ConfigProto()
    config.gpu_options.allow_growth=True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.num_classes, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,
                        FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir+"checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data = len(testX2)
        print("number_of_training_data:", number_of_training_data)
        index=0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for start, end in zip(range(0, number_of_training_data, FLAGS.batch_size), range(FLAGS.batch_size, number_of_training_data+1, FLAGS.batch_size)):
            logits = sess.run(textCNN.logits, feed_dict={textCNN.title: testX2[start:end], textCNN.dropout_keep_prob: 1}) #'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            predicted_labels=get_label_using_logits(logits[0], vocabulary_index2word_label)
            # 7. write question id and labels to file system.
            write_question_id_with_labels(question_id_list[index], predicted_labels, predict_target_file_f)
            index = index+1
        predict_target_file_f.close()


# get label using logits
def get_label_using_logits(logits, vocabulary_index2word_label, top_number=5):
    index_list = np.argsort(logits)[-top_number:]  # print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list = index_list[::-1]
    label_list = []
    for index in index_list:
        label = vocabulary_index2word_label[index]
        label_list.append(label)  # ('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return label_list


# get label using logits
def get_label_using_logits_with_value(logits, vocabulary_index2word_label, top_number=5):
    index_list = np.argsort(logits)[-top_number:]  # print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list = index_list[::-1]
    value_list = []
    label_list = []
    for index in index_list:
        label = vocabulary_index2word_label[index]
        label_list.append(label) # ('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
        value_list.append(logits[index])
    return label_list, value_list


# write question id and labels to file system.
def write_question_id_with_labels(question_id, labels_list, f):
    labels_string = ",".join(labels_list)
    f.write(question_id+","+labels_string+"\n")

if __name__ == "__main__":
    # tf.app.run()
    labels, list_value = get_logits_with_value_by_input(0, 1)
    print("labels:", labels)
    print("list_value:", list_value)