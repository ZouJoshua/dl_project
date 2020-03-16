#!/usr/bin/env bash

export DATA_DIR=/data/work/dl_project/data/corpus/thuc_news
export BERT_BASE_DIR=/data/work/dl_project/data/bert_pre_trained_model/chinese_wwm_ext_tensorflow
export OUTPUT_DIR=/data/work/dl_project/data/model/thuc_news/bert_output

python run_classifier_with_bert.py \
--task_name=thucnews \
--do_train=true \
--do_eval=true \
--data_dir=$DATA_DIR/ \
--vocab_file=$BERT_BASE_DIR/vocab.txt \
--bert_config_file=$BERT_BASE_DIR/bert_config.json \
--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
--max_seq_length=512 \
--train_batch_size=32 \
--learning_rate=2e-5 \
--num_train_epochs=1.0 \
--output_dir=OUTPUT_DIR/