#!/bin/bash

BERT_BASE_DIR="/remote/bones/user/vbalacha/pretrained_bert/uncased_L-12_H-768_A-12"
NQ_BASELINE_DIR="/remote/bones/user/vbalacha/bert-joint-baseline"
#LOAD_MODEL="output/squad-0.1_bertbase_qrystartend_finetune_lr3e-5/best_model"
#SQUAD_DIR="../datasets/zero-shot-relation-extraction/relation_splits/split_both_1/"
APR_DIR="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/files/"
OUTPUT="/remote/bones/user/vbalacha/google-research/fat/fat/fat_bert_nq/generated_files/sharded_kb_data_mc512_unk0.02_test"

mkdir -p $OUTPUT
mkdir -p $OUTPUT/train
mkdir -p $OUTPUT/dev

for i in {2..19}
#for i in 0
do
 echo  $i
 nohup python3 -m fat.fat_bert_nq.prepare_nq_data \
      --is_training=True \
      --verbose_logging=False \
      --split=train \
      --task_id=$i \
      --shard_split_id=0 \
      --input_data_dir=/remote/bones/user/vbalacha/datasets/ent_linked_nq/ \
      --output_data_dir=$OUTPUT \
      --apr_files_dir=$APR_DIR \
      --full_wiki=True \
      --vocab_file=$NQ_BASELINE_DIR/vocab-nq.txt \
      --do_lower_case=True \
      --merge_eval=False \
      --max_context=512 \
      --include_unknown=0.02 > log/$i00.log 2>&1 &
done

wait
echo "All Done"
