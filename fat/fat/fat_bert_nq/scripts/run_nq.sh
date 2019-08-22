#!/bin/bash

BERT_BASE_DIR="/remote/bones/user/vbalacha/pretrained_bert/uncased_L-12_H-768_A-12"
NQ_BASELINE_DIR="/remote/bones/user/vbalacha/bert-joint-baseline"
#LOAD_MODEL="output/squad-0.1_bertbase_qrystartend_finetune_lr3e-5/best_model"
#SQUAD_DIR="../datasets/zero-shot-relation-extraction/relation_splits/split_both_1/"
APR_DIR="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/files/"
DATA="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/data_mc512_unk0.02_test"
LEARNING_RATE=1e-5
NUM_EPOCHS=1
SEED=1
OUTPUT="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/test_output"

mkdir -p $OUTPUT

python -m fat.fat_bert_nq.run_nq \
    --logtostderr \
    --is_training=True \
    --verbose_logging=False \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --output_dir=$OUTPUT \
    --eval_data_path=None \
    --train_precomputed_file=$DATA/train/*.tf-record \
    --train_num_precomputed=494670 \
    --predict_file=None \
    --output_prediction_file=None \
    --do_lower_case=True \
    --max_seg_length=512 \
    --doc_stride=128 \
    --max_query_length=64 \
    --train_batch_size=8 \
    --do_train=True \
    --do_predict=False \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$NUM_EPOCHS \
    --vocab_file=$NQ_BASELINE_DIR/vocab-nq.txt \
    --max_context=512 \
    --include_unknown=0.02 \

