#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author  : Joshua
@Time    : 19-9-10 下午3:31
@File    : predict_main.py
@Desc    : 预测入口
"""

import os

import glob
import json
import tarfile
import time

# import sys
# curr_path = os.path.dirname(os.path.realpath(__file__))
# vc_path = os.path.dirname(curr_path)
# root_path = os.path.dirname(os.path.dirname(vc_path))
# sys.path.append(vc_path)
# sys.path.append(root_path)

import csv
import numpy
import tensorflow as tf

from tensorflow import flags
from tensorflow import gfile
# from tensorflow import logging
import logging


from video_tasks.server import readers
from video_tasks.server import utils

NLP_PREP_MODEL_PATH = "/home/zoushuai/algoproject/nlp_server/src/data/video_tasks/preprocess_model"
NLP_MODEL_PATH = "/home/zoushuai/algoproject/nlp_server/src/data/video_tasks/model"




def load_flags_config():
    pred_flags = flags.FLAGS

    # Feature Extractor flags
    # Optional flags.
    flags.DEFINE_string('extractor_model_dir', NLP_PREP_MODEL_PATH,
                        'Directory to store model files. It defaults to '
                        'src/data/video_tasks/preprocess_model')

    # The following flags are set to match the YouTube-8M dataset format.
    flags.DEFINE_integer('frames_per_second', 1,
                         'This many frames per second will be processed')
    flags.DEFINE_string('labels_feature_key', 'labels',
                        'Labels will be written to context feature with this '
                        'key, as int64 list feature.')
    flags.DEFINE_string('image_feature_key', 'rgb',
                        'Image features will be written to sequence feature with '
                        'this key, as bytes list feature, with only one entry, '
                        'containing quantized feature string.')
    flags.DEFINE_string('video_file_key_feature_key', 'id',
                        'Input <video_file> will be written to context feature '
                        'with this key, as bytes list feature, with only one '
                        'entry, containing the file path of the video. This '
                        'can be used for debugging but not for training or eval.')
    flags.DEFINE_boolean('insert_zero_audio_features', True,
                         'If set, inserts features with name "audio" to be 128-D '
                         'zero vectors. This allows you to use YouTube-8M '
                         'pre-trained model.')

    # Predict Input
    flags.DEFINE_string("train_dir", NLP_MODEL_PATH,
                        "The directory to load the model files from. We assume "
                        "that you have already run eval.py onto this, such that "
                        "inference_model.* files already exist.")
    flags.DEFINE_string(
        "vocabulary_file", os.path.join(os.path.dirname(NLP_MODEL_PATH), "vocabulary.csv"),
        "idxmap file")
    flags.DEFINE_string("input_model_tgz", "",
                        "If given, must be path to a .tgz file that was written "
                        "by this binary using flag --output_model_tgz. In this "
                        "case, the .tgz file will be untarred to "
                        "--untar_model_dir and the model will be used for "
                        "inference.")
    flags.DEFINE_string("untar_model_dir", "",
                        "If --input_model_tgz is given, then this directory will "
                        "be created and the contents of the .tgz file will be "
                        "untarred here.")

    # Predict Output
    # flags.DEFINE_string("output_file", "",
    #                     "The file to save the predictions to.")
    flags.DEFINE_string("output_model_tgz", "",
                        "If given, should be a filename with a .tgz extension, "
                        "the model graph and checkpoint will be bundled in this "
                        "gzip tar. This file can be uploaded to Kaggle for the "
                        "top 10 participants.")
    flags.DEFINE_integer("top_k", 10,
                         "How many predictions to output per video.")

    # Other flags.
    flags.DEFINE_integer(
        "batch_size", 1,
        "How many examples to process per batch.")
    flags.DEFINE_integer("num_readers", 1,
                         "How many threads to use for reading input files.")
    flags.DEFINE_boolean(
        "use_gpu", False,
        "Use device of cpu or gpu. Default False(Use cpu)"
    )
    return pred_flags


def load_idxmap(idxmap_file):
    if not os.path.exists(idxmap_file):
        raise FileNotFoundError("idxmap_file {} not found".format(idxmap_file))
    idxmap = dict()
    with open(idxmap_file, 'r') as csv_file:
        data = csv.DictReader(csv_file, delimiter=",")
        for row in data:
            item = {k: row[k] for k in ('Name', 'Vertical1', 'Vertical2', 'Vertical3')}
            idxmap[row["Index"]] = item
    return idxmap




class Predict(object):

    def __init__(self, pred_flags, logger=None):
        self.pred_flags = pred_flags
        if logger:
            self.log = logger
        else:
            self.log = logging.getLogger("yt8m_video_classification")
            self.log.setLevel(logging.INFO)

        self.reader = self.predict_init()
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.load_model_graph()

    def predict_init(self):
        # logging.set_verbosity(tf.logging.INFO)
        self.log.info("Initing pre_model and reader")
        if self.pred_flags.input_model_tgz:
            if self.pred_flags.train_dir:
                raise ValueError("You cannot supply --train_dir if supplying "
                                 "--input_model_tgz")
            # Untar.
            if not os.path.exists(self.pred_flags.untar_model_dir):
                os.makedirs(self.pred_flags.untar_model_dir)
            tarfile.open(self.pred_flags.input_model_tgz).extractall(self.pred_flags.untar_model_dir)
            self.pred_flags.train_dir = self.pred_flags.untar_model_dir

        flags_dict_file = os.path.join(self.pred_flags.train_dir, "model_flags.json")
        if not os.path.exists(flags_dict_file):
            raise IOError("Cannot find %s. Did you run eval.py?" % flags_dict_file)
        flags_dict = json.loads(open(flags_dict_file).read())

        # convert feature_names and feature_sizes to lists of values
        feature_names, feature_sizes = utils.GetListOfFeatureNamesAndSizes(
            flags_dict["feature_names"], flags_dict["feature_sizes"])

        if flags_dict["frame_features"]:
            reader = readers.YT8MFrameFeatureReader(feature_names=feature_names,
                                                    feature_sizes=feature_sizes,
                                                    prepare_distill=True)
        else:
            reader = readers.YT8MAggregatedFeatureReader(feature_names=feature_names,
                                                         feature_sizes=feature_sizes)

        # if pred_flags.output_file is "":
        #     raise ValueError("'output_file' was not specified. "
        #                      "Unable to continue with inference.")

        # if self.pred_flags.input_data_pattern is "":
        #     raise ValueError("'input_data_pattern' was not specified. "
        #                      "Unable to continue with inference.")
        self.log.info("Successful init pre_model and reader")
        return reader

    def load_model_graph(self):

        config = tf.ConfigProto(allow_soft_placement=True)
        self.sess = tf.Session(config=config)

        checkpoint_file = os.path.join(self.pred_flags.train_dir, "inference_model")
        if not gfile.Exists(checkpoint_file + ".meta"):
            raise IOError("Cannot find %s. Did you run eval.py?" % checkpoint_file)
        meta_graph_location = checkpoint_file + ".meta"
        self.log.info("Loading meta-graph: " + meta_graph_location)

        if self.pred_flags.output_model_tgz:
            with tarfile.open(self.pred_flags.output_model_tgz, "w:gz") as tar:
                for model_file in glob.glob(checkpoint_file + '.*'):
                    tar.add(model_file, arcname=os.path.basename(model_file))
                tar.add(os.path.join(self.pred_flags.train_dir, "model_flags.json"),
                        arcname="model_flags.json")
            self.log.info('Tarred model onto ' + self.pred_flags.output_model_tgz)
        if self.pred_flags.use_gpu:
            with tf.device("/gpu:0"):
                saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
        else:
            with tf.device("/cpu:0"):
                saver = tf.train.import_meta_graph(meta_graph_location, clear_devices=True)
        self.log.info("Restoring variables from " + checkpoint_file)
        saver.restore(self.sess, checkpoint_file)

        self.input_tensor = tf.get_collection("input_batch_raw")[0]
        self.num_frames_tensor = tf.get_collection("num_frames")[0]
        self.predictions_tensor = tf.get_collection("predictions")[0]
        # print(">> input_batch_raw:{}".format(input_tensor))
        # print(">> num_frame_tensor:{}".format(num_frames_tensor))
        # print(">> predictions_tensor:{}".format(predictions_tensor))

        # Workaround for num_epochs issue.
        def set_up_init_ops(variables):
            init_op_list = []
            for variable in list(variables):
                if "train_input" in variable.name:
                    init_op_list.append(tf.assign(variable, 1))
                    variables.remove(variable)
            init_op_list.append(tf.variables_initializer(variables))
            return init_op_list

        self.sess.run(set_up_init_ops(tf.get_collection_ref(
            tf.GraphKeys.LOCAL_VARIABLES)))
        self.log.info("Successful load graph from pre_train model")
        # print(self.sess)
        # print(type(self.sess))


    def format_lines(self, video_ids, predictions, top_k, idxmap):
        batch_size = len(video_ids)
        for video_index in range(batch_size):
            top_indices = numpy.argpartition(predictions[video_index], -top_k)[-top_k:]
            line = [(class_index, predictions[video_index][class_index])
                    for class_index in top_indices]
            line = sorted(line, key=lambda p: -p[1])
            out = dict()
            out["video_file_id"] = video_ids[video_index].decode('utf-8')
            out["video_cats"] = [{"id": str(label), "category": idxmap[str(label)]["Name"], "proba": str(score)} for (label, score) in line]

            yield out



    def predict(self, example_feature_tensor, idxmap):
        with self.graph.as_default():
            self.log.info("Starting predict...")
            self.out = list()
            video_id_batch, video_batch, num_frames_batch = self.get_input_data_tensors(example_feature_tensor, self.pred_flags.batch_size)
            # self.log.info("Getting input tensors...")
            # print(">> video_id_batch:{}".format(video_id_batch))
            # print(">> video_batch:{}".format(video_batch))
            # print(">> num_frame_batch:{}".format(num_frames_batch))

            # self.log.info("Starting queue runner...")
            coord = tf.train.Coordinator()
            threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
            # self.log.info("Successsful start queue runners")
            num_examples_processed = 0
            start_time = time.time()
            try:
                video_id_batch_val, video_batch_val, num_frames_batch_val = self.sess.run(
                    [video_id_batch, video_batch, num_frames_batch])
                predictions_val, = self.sess.run([self.predictions_tensor], feed_dict={self.input_tensor: video_batch_val,
                                                                                  self.num_frames_tensor: num_frames_batch_val})
                now = time.time()
                num_examples_processed += len(video_batch_val)
                num_classes = predictions_val.shape[1]
                self.log.info("Done with inference, num examples processed: " + str(num_examples_processed) + " elapsed seconds: " + "{0:.2f}".format(
                    now - start_time))
                for line in self.format_lines(video_id_batch_val, predictions_val, self.pred_flags.top_k, idxmap):
                    self.out.append(line)
                # self.log.info('done with inference.')
            except Exception as e:
                self.log.error("error with inference {}".format(e))
            coord.request_stop()
            coord.join(threads)
        # return self.out


    def get_input_data_tensors(self,serialized_examples, batch_size, num_readers=1):
        """Creates the section of the graph which reads the input data.

            Args:
            reader: A class which parses the input data.
            data_pattern: A 'glob' style path to the data files.
            batch_size: How many examples to process at a time.
            num_readers: How many I/O threads to use.

            Returns:
            A tuple containing the features tensor, labels tensor, and optionally a
            tensor containing the number of frames per video. The exact dimensions
            depend on the reader being used.

            Raises:
            IOError: If no files matching the given pattern were found.
            """
        with tf.name_scope("input"):
            # files = gfile.Glob(data_pattern)
            # if not files:
            #     raise IOError("Unable to find input files. data_pattern='" +
            #                   data_pattern + "'")
            # logging.info("number of input files: " + str(len(files)))
            # filename_queue = tf.train.string_input_producer(
            #     serialized_examples, num_epochs=1, shuffle=False)
            # tf.add_to_collection("serialized_examples", serialized_examples)
            examples_and_labels = [self.reader.prepare_serialized_examples(serialized_examples)
                                   for _ in range(num_readers)]
            # print(">> len of exmalples_and_labels:{}".format(len(examples_and_labels)))
            # print(">> example_and_labels 类型{}".format(type(examples_and_labels)))
            # print(">> {}".format(examples_and_labels))

            video_id_batch, video_batch, unused_labels, num_frames_batch, batch_preds = (
                tf.train.batch_join(examples_and_labels,
                                    batch_size=batch_size,
                                    allow_smaller_final_batch=True,
                                    enqueue_many=True))
            return video_id_batch, video_batch, num_frames_batch



if __name__ == "__main__":
    from video_classification.server.extract_feature_main import ExtractFeature
    flags = load_flags_config()
    idxmap = load_idxmap("/home/zoushuai/algoproject/nlp_server/src/data/video_tasks/vocabulary.csv")
    ef = ExtractFeature(flags)
    video_file = "/home/zoushuai/Downloads/videoplayback.mp4"
    s = time.time()
    fe = ef.extract(video_file)
    # print(fe)
    e = time.time()
    print(">> 抽取视频特征耗时{}s".format(e - s))
    p = Predict(flags)

    s1 = time.time()
    # app.run(p.predict_with_tfrecord(fe))
    p.predict(fe, idxmap)
    print(p.out[0])
    e1 = time.time()
    print(">> 模型分类耗时{}s".format(e1-s1))
    p.sess.close()
