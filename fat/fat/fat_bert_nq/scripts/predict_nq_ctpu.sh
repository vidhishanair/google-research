#!/bin/bash

#BERT_BASE_DIR="/remote/bones/user/vbalacha/pretrained_bert/uncased_L-12_H-768_A-12"
#BERT_BASE_DIR="gs://cloud-tpu-checkpoints/bert/uncased_L-12_H-768_A-12"
#BERT_BASE_DIR=
#NQ_BASELINE_DIR="/remote/bones/user/vbalacha/bert-joint-baseline"
NQ_BASELINE_DIR=${STORAGE_BUCKET}/bert-joint-baseline

#LOAD_MODEL="output/squad-0.1_bertbase_qrystartend_finetune_lr3e-5/best_model"
#SQUAD_DIR="../datasets/zero-shot-relation-extraction/relation_splits/split_both_1/"
#APR_DIR="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/files/"

DATA="gs://fat_storage/sharded_kb_data_mc512_unk0.02_test"
NQ_DATA="gs://natural_questions/v1.0"
LEARNING_RATE=1e-5
NUM_EPOCHS=1
SEED=1
OUTPUT=$STORAGE_BUCKET/test_output

#mkdir -p $OUTPUT

python3 -m fat.fat_bert_nq.run_nq \
    --logtostderr \
    --is_training=False \
    --verbose_logging=False \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$OUTPUT/model.ckpt-371002 \
    --output_dir=$OUTPUT \
    --eval_data_path=$DATA/dev/*.tf-record \
    --train_precomputed_file=$DATA/train/*.tf-record \
    --train_num_precomputed=494670 \
    --predict_file=$NQ_DATA/dev/*.jsonl.gz \
    --output_prediction_file=$OUTPUT/predictions.json \
    --do_lower_case=True \
    --max_seg_length=512 \
    --doc_stride=128 \
    --max_query_length=64 \
    --train_batch_size=8 \
    --do_train=False \
    --do_predict=True \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$NUM_EPOCHS \
    --vocab_file=$STORAGE_BUCKET/bert-joint-baseline/vocab-nq.txt \
    --max_context=512 \
    --include_unknown=0.02 \
    --use_tpu=True \
    --tpu_name=$TPU_NAME \

