#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-3-14 下午11:33
@File    : textcnn_model.py
@Desc    : TextCNN:
            1. embeddding layers
            2. convolutional layer
            3. max-pooling
            4. softmax layer
"""


import tensorflow as tf


class TextCNN(object):
    def __init__(self,
                 filter_sizes,
                 num_filters,
                 label_size,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 batch_size,
                 sentence_len,
                 vocab_size,
                 embed_size,
                 is_training,
                 clip_gradients=5.0):
        self.label_size = label_size
        self.batch_size = batch_size
        self.sentence_len = sentence_len
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.is_training_flag = is_training
        self.learning_rate = learning_rate
        self.decay_rate = learning_decay_rate
        self.decay_steps = learning_decay_steps

        self.filter_sizes = filter_sizes  # it is a list of int. e.g. [3,4,5]
        self.num_filters = num_filters
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        self.num_filters_total = self.num_filters * len(filter_sizes)  # how many filters totally.
        self.clip_gradients = clip_gradients

        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")  # X
        self.label = tf.placeholder(tf.int32, [None, ], name="label")  # y:[None,label_size]
        # self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.label_size], name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        self.init_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.accuracy = self.acc()


    def init_weights(self):
        """define all weights here"""
        with tf.name_scope("embedding_layer"):
            self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
            self.w = tf.get_variable("w", shape=[self.num_filters_total, self.label_size], initializer=self.initializer)  # [embed_size,label_size], w是随机初始化来的
            self.b = tf.get_variable("b", shape=[self.label_size])       # [label_size]

    def inference(self):
        """
        embedding layers
        convolutional layer
        max-pooling
        softmax layer"""
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.sentence)  # [None,sencente_len,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)  # [None,sencente_len,embed_size,1]

        # if self.use_mulitple_layer_cnn: # this may take 50G memory.
        #    print("use multiple layer CNN")
        #    h=self.cnn_multiple_layers()
        # else: # this take small memory, less than 2G memory
        print("use single layer CNN")
        h = self.cnn_single_layer()
        # 5. logits(use linear layer)and predictions(argmax)
        with tf.name_scope("output"):
            logits = tf.matmul(h, self.w) + self.b  # shape:[None, self.num_classes]==tf.matmul([None,self.embed_size],[self.embed_size,self.num_classes])
        return logits


    def cnn_single_layer(self):
        pooled_outputs = []
        # loop each filter size
        # for each filter, do: convolution-pooling layer, feature shape is 4-d. Feature is a new variable
        # step1.create filters
        # step2.conv (CNN->BN->relu)
        # step3.apply nolinearity(tf.nn.relu)
        # step4.max-pooling(tf.nn.max_pool)
        # step5.dropout
        for i, filter_size in enumerate(self.filter_sizes):
            # with tf.name_scope("convolution-pooling-%s" %filter_size):
            with tf.variable_scope("convolution_pooling_layer_{}".format(filter_size)):
                filter_shape = [filter_size, self.embed_size, 1, self.num_filters]
                # step1.create filter
                filter = tf.get_variable("filter-{}".format(filter_size), filter_shape, initializer=self.initializer)
                # step2.conv operation
                # conv2d ===> computes a 2-D convolution given 4-D `input` and `filter` tensors.
                # *num_filters ---> [1, sentence_len - filter_size + 1, 1, num_filters]
                # *batch_size ---> [batch_size, sentence_len - filter_size + 1, 1, num_filters]
                # conv2d函数的参数：
                # input: [batch, in_height, in_width, in_channels]，
                # filter/kernel: [filter_height, filter_width, in_channels, out_channels]
                # output: 4-D [1,sequence_length-filter_size+1,1,1]，得到的是w.x的部分的值
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="VALID", name="conv-{}".format(filter_size))  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                # conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn_bn_')

                # step3.apply nolinearity
                # h是最终卷积层的输出，即每个feature map，shape = [batch_size, sentence_len - filter_size + 1, 1, num_filters]
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`
                # step4.max-pooling.
                # value: A 4-D `Tensor` with shape `[batch, height, width, channels]
                # ksize: A list of ints that has length >= 4.
                # strides: A list of ints that has length >= 4.
                pooled = tf.nn.max_pool(h, ksize=[1, self.sentence_len - filter_size + 1, 1, 1], strides=[1, 1, 1, 1], padding='VALID', name="pool")  # shape:[batch_size, 1, 1, num_filters].max_pool:performs the max pooling on the input.
                pooled_outputs.append(pooled)
        # step4. combine all pooled features, and flatten the feature.output' shape is a [1,None]
        self.h_pool = tf.concat(pooled_outputs, 3)  # shape:[batch_size, 1, 1, num_filters_total]. tf.concat=>concatenates tensors along one dimension.where num_filters_total=num_filters_1+num_filters_2+num_filters_3
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, self.num_filters_total])  # shape should be:[None,num_filters_total]. here this operation has some result as tf.sequeeze().e.g. x's shape:[3,3];tf.reshape(-1,x) & (3, 3)---->(1,9)

        # step5. add dropout: use tf.nn.dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, keep_prob=self.dropout_keep_prob)  # [None,num_filters_total]
        h = tf.layers.dense(self.h_drop, self.num_filters_total, activation=tf.nn.tanh, use_bias=True)
        return h


    def cnn_multiple_layers(self):
        # loop each filter size
        # for each filter, do: convolution-pooling layer, feature shape is 4-d. Feature is a new variable
        # step1.create filters
        # step2.conv (CNN->BN->relu)
        # step3.apply nolinearity(tf.nn.relu)
        # step4.max-pooling(tf.nn.max_pool)
        # step5.dropout
        pooled_outputs = []
        print("sentence_embeddings_expanded:", self.sentence_embeddings_expanded)
        for i, filter_size in enumerate(self.filter_sizes):
            with tf.variable_scope('cnn_multiple_layers' + "convolution-pooling-%s" % filter_size):
                # Layer1:CONV-RELU
                # 1) CNN->BN->relu
                filter = tf.get_variable("filter-%s" % filter_size, [filter_size, self.embed_size, 1, self.num_filters], initializer=self.initializer)
                conv = tf.nn.conv2d(self.sentence_embeddings_expanded, filter, strides=[1, 1, 1, 1], padding="SAME", name="conv")  # shape:[batch_size,sequence_length - filter_size + 1,1,num_filters]
                conv = tf.contrib.layers.batch_norm(conv, is_training=self.is_training_flag, scope='cnn1')
                print(i, "conv1:", conv)
                b = tf.get_variable("b-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv, b), "relu")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 2) CNN->BN->relu
                h = tf.reshape(h, [-1, self.sentence_len, self.num_filters, 1])  # shape:[batch_size,sequence_length,num_filters,1]
                # Layer2:CONV-RELU
                filter2 = tf.get_variable("filter2-%s" % filter_size, [filter_size, self.num_filters, 1, self.num_filters], initializer=self.initializer)
                conv2 = tf.nn.conv2d(h, filter2, strides=[1, 1, 1, 1], padding="SAME", name="conv2")  # shape:[batch_size,sequence_length-filter_size*2+2,1,num_filters]
                conv2 = tf.contrib.layers.batch_norm(conv2, is_training=self.is_training_flag, scope='cnn2')
                print(i, "conv2:", conv2)
                b2 = tf.get_variable("b2-%s" % filter_size, [self.num_filters])
                h = tf.nn.relu(tf.nn.bias_add(conv2, b2), "relu2")  # shape:[batch_size,sequence_length,1,num_filters]. tf.nn.bias_add:adds `bias` to `value`

                # 3. Max-pooling
                pooling_max = tf.squeeze(tf.nn.max_pool(h, ksize=[1, self.sentence_len, 1, 1],strides=[1, 1, 1, 1], padding='VALID', name="pool"))
                # pooling_avg=tf.squeeze(tf.reduce_mean(h,axis=1)) #[batch_size,num_filters]
                print(i, "pooling:", pooling_max)
                # pooling=tf.concat([pooling_max,pooling_avg],axis=1) #[batch_size,num_filters*2]
                pooled_outputs.append(pooling_max)  # h:[batch_size,sequence_length,1,num_filters]
        # concat
        h = tf.concat(pooled_outputs, axis=1)  # [batch_size,num_filters*len(self.filter_sizes)]
        print("h.concat:", h)

        with tf.name_scope("dropout"):
            h = tf.nn.dropout(h, keep_prob=self.dropout_keep_prob)  # [batch_size,sequence_length - filter_size + 1,num_filters]
        return h  # [batch_size,sequence_length - filter_size + 1,num_filters]

    # def loss_multilabel(self, l2_lambda=0.0001):  # 0.0001
    #     with tf.name_scope("loss"):
    #         # let `x = logits`, `z = labels`.
    #         # The logistic loss is:z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
    #         losses = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.input_y_multilabel, logits=self.logits)
    #         #losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input__y,logits=self.logits)
    #         #losses=-self.input_y_multilabel*tf.log(self.logits)-(1-self.input_y_multilabel)*tf.log(1-self.logits)
    #         print("sigmoid_cross_entropy_with_logits.losses:", losses)
    #         losses = tf.reduce_sum(losses, axis=1)  # shape=(?,). loss for all data in the batch
    #         loss = tf.reduce_mean(losses)         # shape=().   average loss in the batch
    #         l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
    #         loss = loss+l2_losses
    #     return loss


    def loss(self, l2_lambda=0.0001):  # 0.001
        with tf.name_scope("loss"):
            # input: `logits`:[batch_size, num_classes], and `labels`:[batch_size]
            # output: A 1-D `Tensor` of length `batch_size` of the same type as `logits` with the softmax cross entropy loss.
            self.y_true = tf.one_hot(self.label, self.label_size)
            losses = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label, logits=self.logits)
            # sigmoid_cross_entropy_with_logits.
            # losses=tf.nn.softmax_cross_entropy_with_logits(labels=self.input_y, logits=self.logits)
            loss = tf.reduce_mean(losses)
            l2_losses = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables() if 'bias' not in v.name]) * l2_lambda
            loss = loss + l2_losses
        return loss

    def acc(self):
        self.predictions = tf.argmax(self.logits, 1, name="predictions")  # shape:[None,]
        self.y_pred = tf.one_hot(self.predictions, self.label_size)
        correct_prediction = tf.equal(tf.cast(self.predictions, tf.int32), self.label)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")
        return accuracy

    def train_old(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op

    def train(self):
        """based on the loss, use SGD to update parameter"""
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_= learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op