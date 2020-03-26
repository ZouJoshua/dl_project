#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-6-3 下午5:05
@File    : textrnn_model.py
@Desc    : 
"""

import tensorflow as tf

from model_tensorflow.basic_model import BaseModel
import configparser


class Config(object):
    """CNN配置参数"""
    def __init__(self, config_file, section=None):
        config_ = configparser.ConfigParser()
        config_.read(config_file)
        if not config_.has_section(section):
            raise Exception("Section={} not found".format(section))

        self.all_params = {}
        for i in config_.items(section):
            self.all_params[i[0]] = i[1]

        config = config_[section]
        if not config:
            raise Exception("Config file error.")
        self.data_path = config.get("data_path")                           # 数据目录
        self.label2idx_path = config.get("label2idx_path")                 # label映射文件
        self.pretrain_embedding = config.get("pretrain_embedding")         # 预训练词向量文件
        self.stopwords_path = config.get("stopwords_path", "")             # 停用词文件
        self.output_path = config.get("output_path")                       # 输出目录(模型文件\)
        self.ckpt_model_path = config.get("ckpt_model_path", "")           # 模型目录
        self.sequence_length = config.getint("sequence_length")            # 序列长度
        self.num_labels = config.getint("num_labels")                      # 类别数,二分类时置为1,多分类时置为实际类别数
        self.embedding_dim = config.getint("embedding_dim")                # 词向量维度
        self.vocab_size = config.getint("vocab_size")                      # 字典大小
        self.num_filters = config.getint("num_filters")                    # 卷积核数目
        self.hidden_dim = config.getint("hidden_dim")                      # 全连接层神经元
        self.filter_sizes = eval(config.get("filter_sizes", "[3,4,5]"))    # 卷积核尺寸, a list of int. e.g. [3,4,5]
        self.is_training = config.getboolean("is_training", False)
        self.dropout_keep_prob = config.getfloat("dropout_keep_prob")      # 保留神经元的比例
        self.optimization = config.get("optimization", "adam")             # 优化算法
        self.learning_rate = config.getfloat("learning_rate")              # 学习速率
        self.learning_decay_rate = config.getfloat("learning_decay_rate")
        self.learning_decay_steps = config.getint("learning_decay_steps")
        self.l2_reg_lambda = config.getfloat("l2_reg_lambda", 0.0)              # L2正则化的系数，主要对全连接层的参数正则化
        self.max_grad_norm = config.getfloat("max_grad_norm", 5.0)         # 梯度阶段临界值
        self.num_epochs = config.getint("num_epochs")                      # 全样本迭代次数
        self.train_batch_size = config.getint("train_batch_size")          # 训练集批样本大小
        self.eval_batch_size = config.getint("eval_batch_size")            # 验证集批样本大小
        self.test_batch_size = config.getint("test_batch_size")            # 测试集批样本大小
        self.eval_every_step = config.getint("eval_every_step")            # 迭代多少步验证一次模型
        self.model_name = config.get("model_name", "textcnn")              # 模型名称




class RCNNModel(BaseModel):

    def __init__(self, config, vocab_size, word_vectors):
        super(RCNNModel, self).__init__(config=config, vocab_size=vocab_size, word_vectors=word_vectors)

        # 构建模型
        self.build_model()
        # 初始化保存模型的saver对象
        self.init_saver()

    def build_model(self):
        self.embedding_layer()
        self.conv_maxpool_layer()
        # self.multi_conv_maxpool_layer()
        self.full_connection_layer()
        self.cal_loss()

        self.predictions = self.get_predictions()
        # 获得训练入口
        self.train_op, self.summary_op = self.get_train_op()

    def embedding_layer(self):
        """
        词嵌入层
        :return:
        """
        with tf.name_scope("embedding-layer"):
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config.embedding_dim],
                                          initializer=tf.contrib.layers.xavier_initializer())

            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，
            # 维度[batch_size, sequence_length, embedding_dim]
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)
            embedded_words_ = embedded_words

    def


    def build_model(self):

        # 词嵌入层
        with tf.name_scope("embedding"):
            # 利用预训练的词向量初始化词嵌入矩阵
            if self.word_vectors is not None:
                embedding_w = tf.Variable(tf.cast(self.word_vectors, dtype=tf.float32, name="word2vec"),
                                          name="embedding_w")
            else:
                embedding_w = tf.get_variable("embedding_w", shape=[self.vocab_size, self.config["embedding_size"]],
                                              initializer=tf.contrib.layers.xavier_initializer())

            # 利用词嵌入矩阵将输入的数据中的词转换成词向量，
            # 维度[batch_size, sequence_length, embedding_dim]
            embedded_words = tf.nn.embedding_lookup(embedding_w, self.inputs)
            embedded_words_ = embedded_words
            # 卷积操作conv2d的输入是四维[batch_size, sequence_length, embedding_dim, channel],
            # 分别代表着批处理大小、宽度、高度、通道数,因此需要增加维度,设为1,用tf.expand_dims来增大维度

        # 定义两层双向LSTM的模型结构
        with tf.name_scope("Bi-LSTM"):
            for idx, hidden_size in enumerate(self.config["hidden_sizes"]):
                with tf.name_scope("Bi-LSTM" + str(idx)):
                    # 定义前向LSTM结构
                    lstm_fw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)
                    # 定义反向LSTM结构
                    lstm_bw_cell = tf.nn.rnn_cell.DropoutWrapper(
                        tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, state_is_tuple=True),
                        output_keep_prob=self.keep_prob)

                    # 采用动态rnn，可以动态的输入序列的长度，若没有输入，则取序列的全长
                    # outputs是一个元祖(output_fw, output_bw)，其中两个元素的维度都是[batch_size, max_time, hidden_size],
                    # fw和bw的hidden_size一样
                    # self.current_state 是最终的状态，二元组(state_fw, state_bw)，state_fw=[batch_size, s]，s是一个元祖(h, c)
                    outputs, current_state = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell, lstm_bw_cell,
                                                                             embedded_words, dtype=tf.float32,
                                                                             scope="bi-lstm" + str(idx))

                    # 对outputs中的fw和bw的结果拼接 [batch_size, time_step, hidden_size * 2]
                    embedded_words = tf.concat(outputs, 2)

        # 将最后一层Bi-LSTM输出的结果分割成前向和后向的输出
        fw_output, bw_output = tf.split(embedded_words, 2, -1)

        with tf.name_scope("context"):
            shape = [tf.shape(fw_output)[0], 1, tf.shape(fw_output)[2]]
            context_left = tf.concat([tf.zeros(shape), fw_output[:, :-1]], axis=1, name="context_left")
            context_right = tf.concat([bw_output[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        # 将前向，后向的输出和最早的词向量拼接在一起得到最终的词表征
        with tf.name_scope("wordRepresentation"):
            word_representation = tf.concat([context_left, embedded_words_, context_right], axis=2)
            word_size = self.config["hidden_sizes"][-1] * 2 + self.config["embedding_size"]

        with tf.name_scope("text_representation"):
            output_size = self.config["output_size"]
            text_w = tf.Variable(tf.random_uniform([word_size, output_size], -1.0, 1.0), name="text_w")
            text_b = tf.Variable(tf.constant(0.1, shape=[output_size]), name="text_b")

            # tf.einsum可以指定维度的消除运算
            text_representation = tf.tanh(tf.einsum('aij,jk->aik', word_representation, text_w) + text_b)

        # 做max-pool的操作，将时间步的维度消失
        output = tf.reduce_max(text_representation, axis=1)

        # 全连接层的输出
        with tf.name_scope("output"):
            output_w = tf.get_variable(
                "outputW",
                shape=[output_size, self.config["num_classes"]],
                initializer=tf.contrib.layers.xavier_initializer())

            output_b = tf.Variable(tf.constant(0.1, shape=[self.config["num_classes"]]), name="output_b")
            self.l2_loss += tf.nn.l2_loss(output_w)
            self.l2_loss += tf.nn.l2_loss(output_b)
            self.logits = tf.nn.xw_plus_b(output, output_w, output_b, name="logits")
            self.predictions = self.get_predictions()

        self.loss = self.cal_loss()
        self.train_op, self.summary_op = self.get_train_op()





class TextRCNN(object):
    """
    Using LSTM or GRU neural network for text classification
    """
    def __init__(self,
                 sentence_len,
                 label_size,
                 batch_size,
                 hidden_unit,
                 learning_rate,
                 learning_decay_rate,
                 learning_decay_steps,
                 vocab_size,
                 embed_size,
                 is_training,
                 cell='lstm',
                 clip_gradients=5.0):
        self.sentence_len = sentence_len
        self.label_size = label_size
        self.batch_size = batch_size

        self.vocab_size = vocab_size
        self.embed_size = embed_size
        self.hidden_dim = hidden_unit
        self.is_training = is_training
        self.learning_rate = learning_rate

        self.is_training_flag = is_training
        self.learning_rate = learning_rate
        self.decay_rate = learning_decay_rate
        self.decay_steps = learning_decay_steps
        self.gate = cell

        self.clip_gradients = clip_gradients

        self.build_graph()

    def add_placeholders(self):
        self.sentence = tf.placeholder(tf.int32, [None, self.sentence_len], name="sentence")  # X
        self.label = tf.placeholder(tf.int32, [None, ], name="label")  # y:[None,label_size]
        # self.input_y_multilabel = tf.placeholder(tf.float32, [None, self.label_size], name="input_y_multilabel")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")


    def init_weights(self):
        """define all weights here"""
        self.initializer = tf.random_normal_initializer(stddev=0.1)
        with tf.name_scope("embedding_layer"):
            self.embedding = tf.get_variable("embedding", shape=[self.vocab_size, self.embed_size], initializer=self.initializer)  # [vocab_size,embed_size] tf.random_uniform([self.vocab_size, self.embed_size],-1.0,1.0)
        self.w = tf.get_variable("w", shape=[self.hidden_dim, self.label_size], initializer=self.initializer)  # [embed_size,label_size], w是随机初始化来的
        self.b = tf.get_variable("b", shape=[self.label_size])       # [label_size]

    def inference(self):
        """
        embedding layers
        single_hidden_layer
        fully_connection_and_softmax_layer
        """
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.sentence)  # [None,sencente_len,embed_size]
        self.sentence_embeddings_expanded = tf.expand_dims(self.embedded_words, -1)  # [None,sencente_len,embed_size,1]

        print("use rcnn layer")
        h = self.rcnn_layer(self.embedded_words, self.sentence_len, self.hidden_dim, n_hidden_layer=1, if_dropout=True, static=False)

        # 5. logits(use linear layer) and predictions(argmax)
        # full coneection and softmax output
        with tf.variable_scope('fully_connection_layer'):
            # shape:[None, self.label_size]==tf.matmul([None,self.hidden_dim],[self.hidden_dim, self.label_size])
            # logits = tf.nn.softmax(tf.matmul(h, self.w) + self.b, name='logits')
            logits = tf.matmul(h, self.w) + self.b
        return logits

    def get_rnn_cell(self, hidden_unit):
        if self.gate == 'lstm':
            return tf.contrib.rnn.BasicLSTMCell(num_units=hidden_unit, forget_bias=1.0)
        elif self.gate == 'gru':
            return tf.contrib.rnn.GRUCell(num_units=hidden_unit)
        else:
            return tf.contrib.rnn.BasicRNNCell(num_units=hidden_unit)

    def dropout_cell(self, input_cell):
        return tf.contrib.rnn.DropoutWrapper(input_cell, output_keep_prob=self.dropout_keep_prob)

    def hidden_layer(self, hidden_unit, dropout_layer=False, multi_layer=1):

        if multi_layer > 1:
            cells = list()
            for i in range(multi_layer):
                if dropout_layer:
                    cells.append(self.dropout_cell(self.get_rnn_cell(hidden_unit)))
                else:
                    cells.append(self.get_rnn_cell(hidden_unit))
            rnn_cell = tf.contrib.rnn.MultiRNNCell(cells, state_is_tuple=True)
        else:
            rnn_cell = self.get_rnn_cell(hidden_unit)

        return rnn_cell

    def hidden_bi_lstm_layer(self, hidden_unit, dropout_layer=False, multi_layer=1):

        with tf.name_scope("bi-lstm"):

            if multi_layer > 1:
                fw_rnn_cell = list()
                bw_rnn_cell = list()
                for i in range(multi_layer):
                    if dropout_layer:
                        fw_rnn_cell.append(self.dropout_cell(self.get_rnn_cell(hidden_unit)))
                    else:
                        fw_rnn_cell.append(self.get_rnn_cell(hidden_unit))
                for i in range(multi_layer):
                    if dropout_layer:
                        bw_rnn_cell.append(self.dropout_cell(self.get_rnn_cell(hidden_unit)))
                    else:
                        bw_rnn_cell.append(self.get_rnn_cell(hidden_unit))
            else:
                fw_rnn_cell = self.get_rnn_cell(hidden_unit)
                bw_rnn_cell = self.get_rnn_cell(hidden_unit)

        return fw_rnn_cell, bw_rnn_cell


    def rcnn_layer(self, input_x, n_steps, n_hidden_unit, n_hidden_layer=1, if_dropout=False, static=True):
        """
        1，利用Bi-LSTM获得上下文的信息
        2，将Bi-LSTM获得的隐层输出和词向量拼接[fwOutput;wordEmbedding;bwOutput]
        3，将2所得的词表示映射到低维
        4，hidden_size上每个位置的值都取时间步上最大的值，类似于max-pool
        :param input_x: 输入数据
        :param n_steps: 时序
        :param n_hidden_unit: 隐藏层神经元个数
        :param n_hidden_layer: 隐藏层层数
        :param if_dropout: 是否用dropout
        :param static: 是否用动态计算
        :return:
        """
        fw_rnn_cell, bw_rnn_cell = self.hidden_bi_lstm_layer(n_hidden_unit, dropout_layer=if_dropout, multi_layer=n_hidden_layer)
        (output_fw, output_bw), state = tf.nn.bidirectional_dynamic_rnn(cell_fw=fw_rnn_cell, cell_bw=bw_rnn_cell, inputs=input_x,
                                                         dtype=tf.float32)

        with tf.name_scope("context"):
            shape = [tf.shape(output_fw)[0], 1, tf.shape(output_fw)[2]]
            c_left = tf.concat([tf.zeros(shape), output_fw[:, :-1]], axis=1, name="context_left")
            c_right = tf.concat([output_bw[:, 1:], tf.zeros(shape)], axis=1, name="context_right")

        with tf.name_scope("word_representation"):
            y2 = tf.concat([c_left, input_x, c_right], axis=2, name="word_representation")
            embedding_size = 2 * n_hidden_unit + self.embed_size

        # max_pooling层
        with tf.name_scope("max_pooling"):
            fc = tf.layers.dense(y2, self.hidden_dim, activation=tf.nn.relu, name='fc1')
            fc_pool = tf.reduce_max(fc, axis=1)

        return fc_pool


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
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))
        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        train_op = tf.contrib.layers.optimize_loss(self.loss_val, global_step=self.global_step, learning_rate=learning_rate, optimizer="Adam", clip_gradients=self.clip_gradients)
        return train_op

    def train(self):
        """based on the loss, use SGD to update parameter"""
        self.global_step = tf.Variable(0, trainable=False, name="global_step")
        self.epoch_step = tf.Variable(0, trainable=False, name="epoch_step")
        self.epoch_increment = tf.assign(self.epoch_step, tf.add(self.epoch_step, tf.constant(1)))

        learning_rate = tf.train.exponential_decay(self.learning_rate, self.global_step, self.decay_steps, self.decay_rate, staircase=True)
        self.learning_rate_= learning_rate
        optimizer = tf.train.AdamOptimizer(learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(self.loss_val))
        gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.apply_gradients(zip(gradients, variables))
        return train_op

    def build_graph(self):
        self.add_placeholders()
        self.init_weights()
        self.logits = self.inference()
        self.loss_val = self.loss()
        self.train_op = self.train()
        self.accuracy = self.acc()