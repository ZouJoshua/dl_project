#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/20/19 11:09 PM
@File    : lenet5_test.py
@Desc    : lenet5测试集验证准确率

"""


import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_digit_recognition import lenet5_forward
from mnist_digit_recognition import lenet5_backward
import numpy as np
import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
TEST_INTERVAL_SECS = 5


def lenet5_test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, shape=[
            mnist.test.num_examples,
            lenet5_forward.IMAGE_SIZE,
            lenet5_forward.IMAGE_SIZE,
            lenet5_forward.NUM_CHANNELS
        ])

        y_ = tf.placeholder(tf.float32, shape=[None, lenet5_forward.OUTPUT_NODE])
        y = lenet5_forward.forward(x, False, None)

        ema = tf.train.ExponentialMovingAverage(lenet5_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        gpu_config = tf.ConfigProto()
        gpu_config.allow_soft_placement = True
        gpu_config.gpu_options.allow_growth = True

        while True:
            with tf.Session(config=gpu_config) as sess:
                ckpt = tf.train.get_checkpoint_state(lenet5_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split('-')[-1]
                    reshaped_x = np.reshape(mnist.test.images,(
                        mnist.test.num_examples,
                        lenet5_forward.IMAGE_SIZE,
                        lenet5_forward.IMAGE_SIZE,
                        lenet5_forward.NUM_CHANNELS
                    ))
                    accuracy_score = sess.run(accuracy, feed_dict={x: reshaped_x, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return

                time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    lenet5_test(mnist)


if __name__ == "__main__":
    main()
