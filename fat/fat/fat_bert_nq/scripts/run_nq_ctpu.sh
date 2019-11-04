#!/bin/bash

BERT_BASE_DIR="gs://fat_storage/pretrained_bert/wwm_uncased_L-24_H-1024_A-16"
NQ_BASELINE_DIR="gs://fat_storage/bert-joint-baseline"


SEQ_LEN=512
UNK=0.1
TRAIN_NUM=1118021
ALPHA=0.75
#DATA="gs://fat_storage/downweight_incoming_kb_sharded_kb_data_mc48_alpha${ALPHA}_mqseq${SEQ_LEN}_unk${UNK}"
DATA="gs://fat_storage/sharded_kb_data_alpha0.75_mc48_mseq512_unk0.02"
NQ_DATA="gs://natural_questions/v1.0"
LEARNING_RATE=3e-5
NUM_EPOCHS=1
SEED=1
OUTPUT="gs://fat_storage/downweight_incoming_kb_sharded_kb_data_mc48_alpha${ALPHA}_mqseq${SEQ_LEN}_unk${UNK}/output_lr$LEARNING_RATE.epoch$NUM_EPOCHS.seed$SEED.bs32"


#--init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
python3 -m fat.fat_bert_nq.run_nq \
    --logtostderr \
    --is_training=True \
    --verbose_logging=False \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$OUTPUT/model.ckpt-34938 \
    --output_dir=$OUTPUT \
    --eval_data_path=$DATA/dev \
    --train_precomputed_file=$DATA/train/*.tf-record \
    --train_num_precomputed=$TRAIN_NUM \
    --predict_file=$NQ_DATA/dev/*.jsonl.gz \
    --output_prediction_file=$OUTPUT/test_predictions.json \
    --do_lower_case=True \
    --max_seq_length=${SEQ_LEN} \
    --doc_stride=128 \
    --max_query_length=64 \
    --train_batch_size=32 \
    --do_train=False \
    --do_predict=True \
    --learning_rate=$LEARNING_RATE \
    --num_train_epochs=$NUM_EPOCHS \
    --alpha=${ALPHA} \
    --vocab_file=$NQ_BASELINE_DIR/vocab-nq.txt \
    --include_unknowns=${UNK} \
    --use_tpu=True \
    --tpu_name=$TPU_NAME

cd ../../

mkdir tmp

gsutil cp $OUTPUT/test_predictions.json tmp/

python -m natural_questions.nq_eval --logtostderr --gold_path=/home/vbalacha/datasets/ent_linked_nq_new/dev/nq-dev-0???.jsonl.gz --predictions_path=tmp/test_predictions.json --measure_entity_metrics=True --write_pred_analysis --cache_gold_data --prediction_analysis_path=tmp/pred_analysis.tsv > tmp/metrics.json

cat tmp/metrics.json

gsutil cp tmp/metrics.json $OUTPUT/
gsutil cp tmp/pred_analysis.tsv $OUTPUT/

rm -rf tmp
