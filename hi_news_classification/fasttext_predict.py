#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:28
@File    : fasttext_predict.py
@Desc    : process--->
            1.load data(X:list of lint,y:int).
            2.create session. 3.feed data. 4.predict
"""



import tensorflow as tf
import numpy as np
import os
import sys
import codecs

current_work_dir = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_work_dir))
sys.path.append(root_dir)


import warnings
warnings.filterwarnings('ignore')

from tf_model.fasttext_model import FastText
from tflearn.data_utils import pad_sequences
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
from gensim.models import KeyedVectors
from preprocess.preprocess_data_hi import DataSet


data_dir = os.path.join(root_dir, "data", "hi_news")
predict_data_file = os.path.join(data_dir, "corpus_predict")
predict_result_file = os.path.join(data_dir, "corpus_predict_result")
word2vec_file = os.path.join(data_dir, "word2vec.bin")
model_checkpoint = os.path.join(data_dir, "fasttext_checkpoint")
model_saved = os.path.join(data_dir, "pb_model")
cache_file_h5py = os.path.join(data_dir, "data.h5")
cache_file_pickle = os.path.join(data_dir, "vocab_label.pik")
output_dir = os.path.join(data_dir, "summarys")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_string("cache_file_h5py", cache_file_h5py, "path of training/validation/test data.")
tf.flags.DEFINE_string("cache_file_pickle", cache_file_pickle, "path of vocabulary and label files")

tf.flags.DEFINE_integer("label_size", 9, "number of label")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_integer("batch_size", 256, "batch size for training/evaluating")  # 批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 10000, "how many steps before decay learning rate")
tf.flags.DEFINE_float("decay_rate", 0.96, "Rate of decay for learning rate")  # 一次衰减多少
tf.flags.DEFINE_integer("num_sampled", 100, "number of noise sampling")
tf.flags.DEFINE_float("dropout_keep_prob", 0.8, "to control the activation level of neurons")
tf.flags.DEFINE_string("ckpt_dir", model_checkpoint, "checkpoint location for the model")
tf.flags.DEFINE_integer("sentence_len", 200, "max sentence length")
tf.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.flags.DEFINE_boolean("is_training", False, "true:training, false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 20, "epoch times")
tf.flags.DEFINE_integer("validate_every", 1, "validate every validate_every epochs")  # 每1轮做一次验证
tf.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not")
tf.flags.DEFINE_string("predict_source_file", predict_result_file, "target file path for final prediction")
tf.flags.DEFINE_string("predict_target_file", predict_data_file, "target file path for final prediction")


def main(_):
    """导入数据 -> 创建session -> 喂数据 -> 训练 -> (验证) ->（预测）"""

    # step1 -> load data
    ds = DataSet(data_dir, word2vec_file, embedding_dims=FLAGS.embed_size)
    data = ds.load_data_predict(FLAGS.predict_source_file)
    predict_data = list()
    id_list = list()
    vocab_size = len(ds.word2index)
    print("fasttext_model.vocab_size:", vocab_size)
    for doc in data:
        doc_id, pred_data = doc
        id_list.append(doc_id)
        predict_data.append(pred_data)

    # 2.Data preprocessing: Sequence padding
    print("start padding....")
    testX2 = pad_sequences(predict_data, maxlen=FLAGS.sentence_len, value=0.)  # padding to max length
    print("end padding...")

    # 3.create session.
    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        fast_text = FastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.decay_rate, FLAGS.decay_steps,
                             FLAGS.batch_size, FLAGS.num_sampled, FLAGS.dropout_keep_prob,
                             FLAGS.sentence_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)

        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print("Can't find the checkpoint.going to stop")
            return
        # 5.feed data, to get logits
        number_of_training_data = len(testX2)
        print("number_of_predict_data:", number_of_training_data)
        batch_size = 1
        index = 0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data+1, batch_size)):
            logits = sess.run(fast_text.logits, feed_dict={fast_text.sentence: testX2[start:end]})  # 'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            predicted_labels = get_label_using_logits(logits[0], idx2label)
            # 7. write question id and labels to file system.
            write_question_id_with_labels(question_id_list[index], predicted_labels, predict_target_file_f)
            index = index+1
        predict_target_file_f.close()


# get label using logits
def get_label_using_logits(logits, vocabulary_index2word_label, top_number=5):
    # test
    # print("sum_p", np.sum(1.0 / (1 + np.exp(-logits))))
    index_list = np.argsort(logits)[-top_number:]
    index_list = index_list[::-1]
    label_list = []
    for index in index_list:
        label = vocabulary_index2word_label[index]
        label_list.append(label)  # ('get_label_using_logits.label_list:', [u'-3423450385060590478', u'2838091149470021485', u'-3174907002942471215', u'-1812694399780494968', u'6815248286057533876'])
    return label_list


# write question id and labels to file system.
def write_question_id_with_labels(question_id, labels_list, f):
    labels_string = ",".join(labels_list)
    f.write(question_id+","+labels_string + "\n")

if __name__ == "__main__":
    tf.app.run()