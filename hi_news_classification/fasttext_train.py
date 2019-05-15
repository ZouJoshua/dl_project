#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:27
@File    : fasttext_train.py
@Desc    : 

"""

import tensorflow as tf
import numpy as np
import os
from tf_model.fasttext_model import fastText
from tflearn.data_utils import pad_sequences
from gensim.models import KeyedVectors
from preprocess.util import create_vocabulary, load_data, create_label_vocabulary


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("label_size", 15, "number of label")
tf.flags.DEFINE_float("learning_rate", 0.01, "learning rate")
tf.flags.DEFINE_integer("batch_size", 128, "batch size for training/evaluating")  # 批处理的大小 32-->128
tf.flags.DEFINE_integer("decay_steps", 20000, "how many steps before decay learning rate")
tf.flags.DEFINE_float("decay_rate", 0.8, "Rate of decay for learning rate")  # 一次衰减多少
tf.flags.DEFINE_integer("num_sampled", 10, "number of noise sampling")
tf.flags.DEFINE_string("ckpt_dir", "/data/news_classification/fasttext/fasttext_checkpoint/", "checkpoint location for the model")
tf.flags.DEFINE_integer("title_len", 200, "max title length")
tf.flags.DEFINE_integer("embed_size", 300, "embedding size")
tf.flags.DEFINE_boolean("is_training", True, "true:training, false:testing/inference")
tf.flags.DEFINE_integer("num_epochs", 15, "epoch times")
tf.flags.DEFINE_integer("validate_every", 10, "validate every validate_every epochs")  # 每10轮做一次验证
tf.flags.DEFINE_boolean("use_embedding", True, "whether to use embedding or not")
# tf.app.flags.DEFINE_string("cache_path", "fast_text_checkpoint/data_cache.pik", "checkpoint location for the model")

content_word_vector_path = "/data/news_content/content_word_vector.bin"
model_path = "/data/news_content/word2vec_content_model"


def main(_):
    """导入数据 -> 创建session -> 喂数据 -> 训练 -> (验证) ->（预测）"""

    vocabulary_word2idx, vocabulary_idx2word = create_vocabulary(model_path)
    vocab_size = len(vocabulary_word2idx)
    label2idx, _ = create_label_vocabulary()
    train, test, _ = load_data(vocabulary_word2idx, label2idx)
    trainX, trainY = train
    testX, testY = test
    print("testX.shape:", np.array(testX).shape)  # 2500个list.每个list代表一句话
    print("testY.shape:", np.array(testY).shape)  # 2500个label
    print("testX[0]:", testX[0])  # [17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
    print("testX[1]:", testX[1])
    print("testY[0]:", testY[0])  # 0
    print("testY[1]:", testY[1])   # 0

    # Sequence padding
    print("start padding ...")
    trainX = pad_sequences(trainX, maxlen=FLAGS.title_len, value=0.)  # padding to max length
    testX = pad_sequences(testX, maxlen=FLAGS.title_len, value=0.)    # padding to max length
    ###############################################################################################
    # with open(FLAGS.cache_path, 'w') as data_f: #save data to cache file, so we can use it next time quickly.
    #    pickle.dump((trainX,trainY,testX,testY,vocabulary_index2word),data_f)
    ###############################################################################################
    print("testX[0]:", testX[0])
    print("testX[1]:", testX[1])  # [17, 25, 10, 406, 26, 14, 56, 61, 62, 323, 4]
    print("testY[0]:", testY[0])  # 0
    print("testY[1]:", testY[1])  # 0
    print("end padding ...")

    config = tf.ConfigProto()
    # config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata
        # Instantiate Model
        fast_text = fastText(FLAGS.label_size, FLAGS.learning_rate, FLAGS.batch_size, FLAGS.decay_steps, FLAGS.decay_rate,
                             FLAGS.num_sampled, FLAGS.title_len, vocab_size, FLAGS.embed_size, FLAGS.is_training)
        # Initialize Save
        saver = tf.train.Saver()
        if os.path.exists(FLAGS.ckpt_dir + "checkpoint"):
            print("Restoring variables from checkpoint")
            saver.restore(sess, tf.train.latest_checkpoint(FLAGS.ckpt_dir))
        else:
            print('Initializing Variables')
            sess.run(tf.global_variables_initializer())
            if FLAGS.use_embedding:  # load pre-trained word vectors
                assign_pretrained_word_embedding(sess, vocabulary_idx2word, vocab_size, fast_text)

        curr_epoch = sess.run(fast_text.epoch_step)
        # tl = timeline.Timeline(run_metadata.step_stats)
        # feed data and train the model
        number_of_training_data = len(trainX)
        batch_size = FLAGS.batch_size
        for epoch in range(curr_epoch, FLAGS.num_epochs):  # range(start,stop,step_size)
            loss, acc, counter = 0.0, 0.0, 0
            for start, end in zip(range(0, number_of_training_data, batch_size), range(batch_size, number_of_training_data, batch_size)):
                if epoch == 0 and counter == 0:
                    print("trainX[start:end]:", trainX[start:end])
                    print("trainY[start:end]:", trainY[start:end])
                curr_loss, curr_acc, _ = sess.run([fast_text.loss_val, fast_text.accuracy, fast_text.train_op],
                                                  feed_dict={fast_text.title: trainX[start:end], fast_text.label: trainY[start:end]})
                loss, acc, counter = loss+curr_loss, acc+curr_acc, counter+1
                if counter % 500 == 0:
                    print("Epoch %d\tBatch %d\tTrain Loss:%.3f\tTrain Accuracy:%.3f" %(epoch, counter, loss/float(counter), acc/float(counter)))
            # epoch increment
            print("going to increment epoch counter....")
            sess.run(fast_text.epoch_increment)

            print(epoch, FLAGS.validate_every, (epoch % FLAGS.validate_every == 0))
            if epoch % FLAGS.validate_every == 0:
                eval_loss, eval_acc = do_eval(sess, fast_text, testX, testY, batch_size)
                print("Epoch %d Validation Loss:%.3f\tValidation Accuracy: %.3f" % (epoch, eval_loss, eval_acc))
                # save model to checkpoint
                save_path = FLAGS.ckpt_dir + "model.ckpt"
                saver.save(sess, save_path, global_step=fast_text.epoch_step)

        # test model performance on test dataset and report test accuracy
        test_loss, test_acc = do_eval(sess, fast_text, testX, testY, batch_size)
        print("test loss: %2.4f, test accruacy: %2.4f" % (test_loss, test_acc))
    pass


def assign_pretrained_word_embedding(sess, vocabulary_idx2word, vocab_size, fast_text):
    print("start using pre-trained word embedding...")
    # word2vecc=word2vec.load('word_embedding.txt') #load vocab-vector fiel.word2vecc['w91874']
    word2vec_model = KeyedVectors.load_word2vec_format(content_word_vector_path, binary=True)
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
    t_assign_embedding = tf.assign(fast_text.embedding, word_embedding)  # assign this value to our embedding variables of our model.
    print("wordidx 2 vec finished...")
    sess.run(t_assign_embedding)
    print("word. exists embedding:", count_exist, " ;word not exist embedding:", count_not_exist)
    print("using pre-trained word emebedding ended...")


# Do validation on validate dataset and report loss、accuracy
def do_eval(sess, fast_text, evalX, evalY, batch_size):
    number_examples = len(evalX)
    eval_loss, eval_acc, eval_counter = 0.0, 0.0, 0
    for start, end in zip(range(0, number_examples, batch_size), range(batch_size, number_examples, batch_size)):
        curr_eval_loss, curr_eval_acc, = sess.run([fast_text.loss_val, fast_text.accuracy],
                                          feed_dict={fast_text.title: evalX[start:end], fast_text.label: evalY[start:end]})
        eval_loss, eval_acc, eval_counter = eval_loss+curr_eval_loss, eval_acc+curr_eval_acc, eval_counter+1
    return eval_loss/float(eval_counter), eval_acc/float(eval_counter)

if __name__ == "__main__":
    tf.app.run()
