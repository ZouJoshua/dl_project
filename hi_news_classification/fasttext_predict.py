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
from tf_model.fasttext_model import fastText
from tflearn.data_utils import pad_sequences
import os
import codecs
from preprocess.preprocess_data_hi import DataSet


current_work_dir = os.path.realpath(__file__)
root_dir = os.path.dirname(os.path.dirname(current_work_dir))
data_dir = os.path.join(root_dir, "data", "hi_news")
training_data_file = os.path.join(data_dir, "top_category_corpus")
word2vec_file = os.path.join(data_dir, "word2vec.bin")
model_checkpoint = os.path.join(data_dir, "fasttext_checkpoint")
model_saved = os.path.join(data_dir, "fasttext_model_saved")
cache_file_h5py = os.path.join(data_dir, "data.h5")
cache_file_pickle = os.path.join(data_dir, "vocab_label.pik")
output_dir = os.path.join(data_dir, "summarys")


FLAGS=tf.flags.FLAGS
tf.flags.DEFINE_integer("label_size", 9, "number of label")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_integer("batch_size", 512, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 5000, "how many steps before decay learning rate.")  # 批处理的大小 32-->128
tf.flags.DEFINE_float("decay_rate", 0.5, "Rate of decay for learning rate.")
tf.flags.DEFINE_integer("num_sampled", 10, "number of noise sampling")
tf.flags.DEFINE_string("ckpt_dir", "data/news_classification/fasttext/fasttext_checkpoint/", "checkpoint location for the model")
tf.flags.DEFINE_integer("title_len", 300, "max title length")
tf.flags.DEFINE_integer("embed_size", 100, "embedding size")
tf.flags.DEFINE_boolean("is_training", False, "is traning.true:tranining,false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 15, "embedding size")
tf.flags.DEFINE_integer("validate_every", 3, "Validate every validate_every epochs.")  # 每3轮做一次验证
tf.flags.DEFINE_string("predict_target_file", "data/news_classification/fasttext/fasttext_checkpoint/predict_target", "target file path for final prediction")
tf.flags.DEFINE_string("predict_source_file", 'data/news_classification/fasttext/fasttext_checkpoint/predict_source', "target file path for final prediction")


def main(_):
    """导入数据 -> 创建session -> 喂数据 -> 训练 -> (验证) ->（预测）"""
    ds = DataSet(data_dir, word2vec_file, training_data_file, embedding_dims=FLAGS.embed_size)

    vocab_size = len(ds.word2index)
    print("fasttext_model.vocab_size:", vocab_size)
    test = load_data_predict(vocabulary_word2idx, label2idx, questionid_question_lists)
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
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        # 4.Instantiate Model
        fast_text = fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.decay_rate, FLAGS.decay_steps,
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
        print("number_of_training_data:", number_of_training_data)
        batch_size = 1
        index = 0
        predict_target_file_f = codecs.open(FLAGS.predict_target_file, 'a', 'utf8')
        for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data+1, batch_size)):
            logits = sess.run(fast_text.logits, feed_dict={fast_text.title: testX2[start:end]})  # 'shape of logits:', ( 1, 1999)
            # 6. get lable using logtis
            predicted_labels = get_label_using_logits(logits[0],idx2label)
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