#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 11/18/19 11:41 PM
@File    : nn_test.py
@Desc    : mnist_forward mnist_backward 测试

"""

import time
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from mnist_digit_recognition import nn_forward
from mnist_digit_recognition import nn_backward


TEST_INTERVAL_SECS = 5

def mnist_test(mnist):
    with tf.Graph().as_default() as g:
        x = tf.placeholder(tf.float32, [None, nn_forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, nn_forward.OUTPUT_NODE])

        y = nn_forward.forward(x, None)

        ema = tf.train.ExponentialMovingAverage(nn_backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)

        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                ckpt = tf.train.get_checkpoint_state(nn_backward.MODEL_SAVE_PATH)
                if ckpt and ckpt.model_checkpoint_path:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    global_step = ckpt.model_checkpoint_path.split("/")[-1].split('-')[-1]
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s training step(s), test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return

            time.sleep(TEST_INTERVAL_SECS)

def main():
    mnist = input_data.read_data_sets("./MNIST_data", one_hot=True)
    mnist_test(mnist)

if __name__ == "__main__":
    main()
