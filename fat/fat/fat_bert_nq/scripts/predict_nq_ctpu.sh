#!/bin/bash

BERT_BASE_DIR="gs://fat_storage/pretrained_bert/wwm_uncased_L-24_H-1024_A-16"
NQ_BASELINE_DIR="gs://fat_storage/bert-joint-baseline"

SEQ_LEN=512
UNK=0.02
DATA="gs://fat_storage/sharded_kb_data_mc48_mseq${SEQ_LEN}_unk${UNK}"
NQ_DATA="gs://natural_questions/v1.0"
LEARNING_RATE=3e-5
NUM_EPOCHS=1
SEED=1
OUTPUT="gs://fat_storage/sharded_kb_data_mc48_mseq${SEQ_LEN}_unk${UNK}/output_lr$LEARNING_RATE.epoch$NUM_EPOCHS.seed$SEED.bs32"


python3 -m fat.fat_bert_nq.run_nq \
    --logtostderr \
    --is_training=False \
    --verbose_logging=False \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$OUTPUT/model.ckpt-19000 \
    --output_dir=$OUTPUT \
    --eval_data_path=$DATA/dev \
    --train_precomputed_file=$DATA/train/*.tf-record \
    --train_num_precomputed=203868 \
    --predict_file=$NQ_DATA/dev/*.jsonl.gz \
    --output_prediction_file=$OUTPUT/predictions.json \
    --do_lower_case=True \
    --max_seq_length=${SEQ_LEN} \
    --doc_stride=128 \
    --max_query_length=64 \
    --train_batch_size=8 \
    --do_train=False \
    --do_predict=True \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$NUM_EPOCHS \
    --vocab_file=$NQ_BASELINE_DIR/vocab-nq.txt \
    --include_unknowns=${UNK} \
    --use_tpu=True \
    --tpu_name=$TPU_NAME 

mkdir tmp

gsutil cp $OUTPUT/predictions.json tmp/

python -m natural_questions.nq_eval --logtostderr --gold_path=/home/vidhishabalachandran/datasets/v1.0/dev/nq-dev-0?.jsonl.gz --predictions_path=tmp/predictions.json > tmp/metrics.json

cat tmp/metrics.json

gsutil cp tmp/metrics.json $OUTPUT/

rm -rf tmp
