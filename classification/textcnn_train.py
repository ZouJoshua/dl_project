#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:33
@File    : textcnn_train.py
@Desc    : 

"""


import classification as tf
import numpy as np
from tf_model.textcnn_model import TextCNN
from preprocess.util import create_vocabulary, create_label_vocabulary, load_data
from tflearn.data_utils import pad_sequences
import os
from gensim.models import KeyedVectors
from tensorboard import summary as summary_lib


FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer("label_size", 15, "number of label")
tf.app.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.app.flags.DEFINE_integer("batch_size", 128, "Batch size for training/evaluating.")  # 批处理的大小 32-->128
tf.app.flags.DEFINE_integer("decay_steps", 6000, "how many steps before decay learning rate.")
tf.app.flags.DEFINE_float("decay_rate", 0.65, "Rate of decay for learning rate.")  # 0.65一次衰减多少
tf.app.flags.DEFINE_string("ckpt_dir", "data/textcnn/textcnn_title_checkpoint/", "checkpoint location for the model")
tf.app.flags.DEFINE_integer("title_len", 200, "max title length")
tf.app.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.app.flags.DEFINE_boolean("is_training", True, "is traning.true:tranining,false:testing/inference")
tf.app.flags.DEFINE_integer("num_epochs", 15, "number of epochs to run.")
tf.app.flags.DEFINE_integer("validate_every", 10, "Validate every validate_every epochs.")  # 每10轮做一次验证
tf.app.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not.")
tf.app.flags.DEFINE_integer("num_filters", 256, "number of filters")  # 256--->512

filter_sizes = [3, 4, 5]  # todo: 找最佳的filter size
title_word_vector_path = "data/news_content/preprocess/title_word_vector.bin"
model_path = "data/news_content/preprocess/word2vec_title_model"


def main(_):
    vocabulary_word2idx, vocabulary_idx2word = create_vocabulary(model_path, name_scope="cnn2")
    vocab_size = len(vocabulary_word2idx)
    print("cnn_model.vocab_size:", vocab_size)
    label2idx, idx2label = create_label_vocabulary(name_scope="cnn2")
    train, test, _ = load_data(vocabulary_word2idx, label2idx)
    trainX, trainY = train
    testX, testY = test

    print("start padding...")
    trainX = pad_sequences(trainX, maxlen=FLAGS.title_len, value=0.)
    testX = pad_sequences(testX, maxlen=FLAGS.title_len, value=0.)
    # with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
    #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
    print("trainX[0]:", trainX[0])  # ;print("trainY[0]:", trainY[0])
    print("end padding...")

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        textCNN = TextCNN(filter_sizes, FLAGS.num_filters, FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps,
                        FLAGS.decay_rate, FLAGS.title_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)

        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring Variables from Checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:
                assign_pretrained_word_embedding(sess, vocabulary_idx2word, vocab_size, textCNN)
        curr_epoch = sess.run(textCNN.epoch_step)

        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])
                    print("trainY[start:end]:", trainY[start:end])
                feed_dict = {textCNN.title: trainX[start:end], textCNN.dropout_keep_prob: 0.5, textCNN.label: trainY[start:end]}
                curr_loss, curr_acc, _ = sess.run([textCNN.loss_val, textCNN.accuracy, textCNN.train_op],
                                                  feed_dict=feed_dict)
                loss, counter, acc = loss+curr_loss, counter+1, acc+curr_acc
                if counter % 50 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" % (epoch, counter, loss/float(counter), acc/float(counter)))

            # epoch increment
            print("going to increment epoch counter....")
            sess.run(textCNN.epoch_increment)

            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, textCNN, testX, testY, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=epoch)

        test_loss, test_acc = do_eval(sess, textCNN, testX, testY, batch_size)
        print("test loss: %2.4f, test accruacy: %2.4f" % (test_loss, test_acc))
    pass


def assign_pretrained_word_embedding(sess, vocabulary_idx2word, vocab_size, textCNN):
    print("start using pre-trained word embedding...")
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = KeyedVectors.load_word2vec_format(title_word_vector_path, binary=True)
    word2vec_dict = {}
    for word, vector in zip(word2vec_model.vocab, word2vec_model.vectors):
        if word.isalpha():
            word2vec_dict[word] = vector
    word_embedding_2dlist = [[]] * vocab_size  # 用于存储词id对应的embedding
    word_embedding_2dlist[0] = np.zeros(FLAGS.embed_size)  # assign empty for first word:'PAD'
    bound = np.sqrt(6.0) / np.sqrt(vocab_size)  # bound for random variables.
    count_exist = 0
    count_not_exist = 0

    for i in range(1, vocab_size):  # loop each word
        word = vocabulary_idx2word[i]  # get a word
        try:
            embedding = word2vec_dict[word]  # try to get vector:it is an array.
        except Exception:
            embedding = None
        if embedding is not None:  # the 'word' exist a embedding
            word_embedding_2dlist[i] = embedding
            count_exist = count_exist + 1  # assign array to this word.
        else:
            word_embedding_2dlist[i] = np.random.uniform(-bound, bound, FLAGS.embed_size)
            count_not_exist = count_not_exist + 1  # init a random value for the word.
    word_embedding_final = np.array(word_embedding_2dlist)  # covert to 2d array.
    word_embedding = tf.convert_to_tensor(word_embedding_final, dtype=tf.float32)  # convert to tensor
    t_assign_embedding = tf.assign(textCNN.embedding, word_embedding)
    print("wordidx 2 vec finished...")
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding ended...")


# 在验证集上做验证，报告损失、精确度
def do_eval(sess, textCNN, evalX, evalY, batch_size):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        feed_dict = {textCNN.title: evalX[start:end], textCNN.dropout_keep_prob: 1, textCNN.label: evalY[start:end]}
        curr_eval_loss, logits, curr_eval_acc = sess.run([textCNN.loss_val, textCNN.logits, textCNN.accuracy],
                                                         feed_dict=feed_dict)
        eval_loss, eval_acc, eval_counter = eval_loss+curr_eval_loss, eval_acc+curr_eval_acc, eval_counter+1
    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)


if __name__ == "__main__":
    tf.app.run()